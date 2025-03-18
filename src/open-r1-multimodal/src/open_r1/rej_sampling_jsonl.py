# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
import base64
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from utils.math import compute_score

from math_verify import parse, verify
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import threading

def get_line_count(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def evaluate_answer_similarity(student_answer, ground_truth):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": "You are a evaluation expert. First, analyze the student's response to identify and extract their final answer. Then, compare the extracted answer with the correct solution. Output ONLY '1.0' if the extracted answer matches the correct solution in meaning, or '0.0' if the student's response does not contain a clear or correct answer. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student's response: {student_answer}\nCorrect solution: {ground_truth}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0

def llm_reward(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)

def mcq_reward(content, sol, **kwargs):
    # For multiple choice, extract and compare choices
    has_choices = extract_choice(sol)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r'(yes|no)', ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ''
    student_yes_no = re.search(r'(yes|no)', student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ''

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0

    return reward

def calculate_map(pred_bbox_list, gt_bbox_list):
    # Calculate mAP

    # Initialize COCO object for ground truth
    gt_json = {"annotations": [], "images": [], "categories": []}
    gt_json["images"] = [{
        "id": 0,
        "width": 2048,
        "height": 2048,
        "file_name": "image_0.jpg"
    }]

    gt_json["categories"] = []

    cats2id = {}
    cat_count = 0
    for idx, gt_bbox in enumerate(gt_bbox_list):
        if gt_bbox["label"] not in cats2id:
            cats2id[gt_bbox["label"]] = cat_count
            gt_json["categories"].append({
                "id": cat_count,
                "name": gt_bbox["label"]
            })
            cat_count += 1
        
        gt_json["annotations"].append({
            "id": idx+1,
            "image_id": 0,
            "category_id": cats2id[gt_bbox["label"]],
            "bbox": [gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][1], gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0], gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]],
            "area": (gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0]) * (gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1]),
            "iscrowd": 0
        })
    coco_gt = COCO(gt_json)

    dt_json = []
    for idx, pred_bbox in enumerate(pred_bbox_list):
        try:
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": [pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][1], pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0], pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1]],
                "score": 1.0,
                "area": (pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0]) * (pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1])
            })
        except:
            pass
    
    if len(dt_json) == 0:
        return 0.0
    
    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[1]

def map_reward(content, sol, **kwargs):
    """
    Calculate mean average precision (mAP) reward between predicted and ground truth bounding boxes
    
    Args:
        content: String containing predicted bounding boxes in JSON format
        sol: String containing ground truth bounding boxes in JSON format
        
    Returns:
        float: mAP reward score between 0 and 1
    """
    # Extract JSON content between ```json tags
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, sol, re.DOTALL)
    bbox_json = json_match.group(1).strip() if json_match else None

    # Parse ground truth JSON to get bbox list
    gt_bbox_list = []
    if bbox_json:
        bbox_data = json.loads(bbox_json)
        gt_bbox_list = [item for item in bbox_data]
    
    # Parse predicted JSON to get bbox list
    pred_bbox_list = []
    json_match = re.search(pattern, content, re.DOTALL)
    if json_match:
        try:
            bbox_data = json.loads(json_match.group(1).strip())
            pred_bbox_list = [item for item in bbox_data]
        except:
            # Return empty list if JSON parsing fails
            pred_bbox_list = []

    # Calculate mAP if both prediction and ground truth exist
    if len(pred_bbox_list) > 0 and len(gt_bbox_list) > 0:
        bbox_reward = calculate_map(pred_bbox_list, gt_bbox_list)
    else:
        bbox_reward = 0.0
    
    return bbox_reward


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None
def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)
def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
        # Extract answer from solution if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try: 
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)
            
            has_yes_no = bool(re.search(r'\b(yes|no)\b', ground_truth.lower()))
            
            if has_numbers:
                # For numeric answers, use exact matching
                reward = math_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            elif has_yes_no:
                reward = yes_no_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        elif accu_reward_method == 'yes_no':
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == 'llm':
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'map':
            reward = map_reward(content, sol)
        elif accu_reward_method == 'math':
            reward = math_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)  
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")     

        
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def rejection_sampling_vqa(
    image_paths, 
    questions, 
    ground_truths, 
    output_file_path, 
    openai_api_key=None, 
    openai_base_url=None, 
    generation_kwargs=None,
    reward_funcs=None,
    accu_reward_methods=None,
    reward_threshold=0.5,
    original_image_paths=None,
    line_numbers=None,
    num_of_generations=1,
):
    """
    Perform rejection sampling for VQA data generation using OpenAI's vision models.
    
    Args:
        image_paths (list): List of paths to images
        questions (list): List of questions corresponding to images
        ground_truths (list): List of ground truth answers
        output_file_path (str): Path to save the output JSONL file
        openai_api_key (str, optional): OpenAI API key
        openai_base_url (str, optional): OpenAI API base URL
        generation_kwargs (dict, optional): Additional kwargs for generation
        reward_funcs (list, optional): List of reward functions to use
        accu_reward_methods (list, optional): List of accuracy reward methods
        original_image_paths (list, optional): List of original image paths from input JSONL
        line_numbers (list, optional): List of line numbers from the input file
        num_of_generations (int, optional): Number of generations to try for each example
    
    Returns:
        int: Number of examples that passed rejection sampling
    """
    if not reward_funcs:
        reward_funcs = [accuracy_reward, format_reward]
    
    if not accu_reward_methods:
        accu_reward_methods = ["default"] * len(questions)
    
    if not line_numbers:
        line_numbers = list(range(len(questions)))
    
    # Configure OpenAI client
    import openai
    openai.base_url = openai_base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "sk-proj-1234567890")
    
    # Create OpenAI client
    global client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url
    )
    
    # get the model name from the generation_kwargs
    model_name = generation_kwargs.get("model", "gpt-4o-mini")

    # Default generation parameters
    default_kwargs = {
        "model": model_name,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # Update with user-provided kwargs
    if generation_kwargs:
        default_kwargs.update(generation_kwargs)
    
    accepted_count = 0
    
    # Create a log directory for detailed logs
    log_dir = os.path.join(os.path.dirname(output_file_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rejection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Log configuration
    with open(log_file, 'w') as log_f:
        log_f.write(f"Rejection Sampling Configuration:\n")
        log_f.write(f"Model: {default_kwargs['model']}\n")
        log_f.write(f"Temperature: {default_kwargs['temperature']}\n")
        log_f.write(f"Max Tokens: {default_kwargs['max_tokens']}\n")
        log_f.write(f"Reward Threshold: {reward_threshold}\n")
        log_f.write(f"Number of Generations: {num_of_generations}\n\n")
    
    # Create or open the output file
    with open(output_file_path, 'a', encoding='utf-8') as f:
        # Process each example
        for idx, (image_path, question, ground_truth, line_number) in enumerate(zip(image_paths, questions, ground_truths, line_numbers)):
            try:
                # Log progress
                with open(log_file, 'a') as log_f:
                    log_f.write(f"\n--- Processing example {idx+1}/{len(questions)}, Line {line_number} ---\n")
                    log_f.write(f"Question: {question}\n")
                    log_f.write(f"Ground Truth: {ground_truth}\n")
                    log_f.write(f"Image Path: {image_path}\n")
                
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}, skipping example {idx+1}/{len(questions)}, Line {line_number}")
                    with open(log_file, 'a') as log_f:
                        log_f.write(f"Error: Image not found, skipping\n")
                    continue
                
                # Load and encode image for API
                try:
                    def encode_image(image_path):
                        with open(image_path, "rb") as image_file:
                            return base64.b64encode(image_file.read()).decode('utf-8')
                    
                    base64_image = encode_image(image_path)
                except Exception as e:
                    print(f"Error encoding image {image_path}: {e}, skipping example {idx+1}/{len(questions)}, Line {line_number}")
                    with open(log_file, 'a') as log_f:
                        log_f.write(f"Error encoding image: {e}\n")
                    continue
                
                # Create prompt with instruction to use think/answer format
                prompt = question + " Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                
                # Try multiple generations
                accepted = False

                try:
                    with open(log_file, 'a') as log_f:
                        log_f.write(f"\nGenerating {num_of_generations} completions in a single API call\n")
                    
                    # Use the 'n' parameter to generate multiple completions in one API call
                    response = client.chat.completions.create(
                        model=default_kwargs["model"],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=default_kwargs.get("max_tokens", 1024),
                        temperature=default_kwargs.get("temperature", 0.7),
                        top_p=default_kwargs.get("top_p", 1.0),
                        n=num_of_generations  # Generate multiple completions
                    )
                    
                    # Process each generated completion
                    for gen_idx, choice in enumerate(response.choices):
                        # Extract the model's response
                        model_response = choice.message.content
                        
                        with open(log_file, 'a') as log_f:
                            log_f.write(f"Model Response {gen_idx+1}/{num_of_generations}:\n{model_response}\n\n")
                        
                        # Format for reward evaluation
                        completion = [[{"content": model_response}]]
                        solution = [f"<answer> {ground_truth} </answer>"]
                        
                        # Evaluate with reward functions
                        rewards = []
                        for reward_func in reward_funcs:
                            if reward_func == accuracy_reward:
                                reward = reward_func(completion, solution, accu_reward_method=[accu_reward_methods[idx]], 
                                                    image_path=[image_path], problem=[question])
                            else:
                                reward = reward_func(completion)
                            rewards.append(reward[0])
                        
                        with open(log_file, 'a') as log_f:
                            log_f.write(f"Rewards for completion {gen_idx+1}: {rewards}\n")
                        
                        # Check if this generation passes all reward thresholds
                        if all(r > reward_threshold for r in rewards):
                            # Create data object
                            data_obj = {
                                "image": original_image_paths[idx] if original_image_paths else image_path,
                                "conversations": [
                                    {"role": "user", "value": f"<image>\n{prompt}"},
                                    {"role": "assistant", "value": model_response}
                                ],
                                "question": question,
                                "solution": ground_truth,
                                "accu_reward_method": accu_reward_methods[idx],
                                "rewards": rewards,
                                "passed_rejection_sampling": True,
                                "input_line_number": line_number,  # Store the line number for resuming
                                "generation_number": gen_idx + 1,  # Store which generation attempt succeeded
                                "total_generations": num_of_generations,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Write to output file without thread lock
                            f.write(json.dumps(data_obj, ensure_ascii=False) + "\n")
                            f.flush()
                            accepted_count += 1
                            accepted = True
                            
                            with open(log_file, 'a') as log_f:
                                log_f.write(f"ACCEPTED: Example passed rejection sampling (completion {gen_idx+1})\n")
                            
                            # Print without thread lock
                            print(f"Processed {idx+1}/{len(questions)}, Line {line_number}, Accepted: {accepted_count}, Generation: {gen_idx+1}/{num_of_generations}, Rewards: {rewards}")
                            break  # Stop processing further completions once we find an acceptable one
                        else:
                            with open(log_file, 'a') as log_f:
                                log_f.write(f"REJECTED: Completion {gen_idx+1} - Rewards below threshold\n")
                            print(f"Processed {idx+1}/{len(questions)}, Line {line_number}, Generation {gen_idx+1}/{num_of_generations} Rejected, Rewards: {rewards}")
                    
                    # If all generations failed, log the final result
                    if not accepted:
                        with open(log_file, 'a') as log_f:
                            log_f.write(f"FINAL RESULT: All {num_of_generations} completions rejected\n")
                        print(f"Processed {idx+1}/{len(questions)}, Line {line_number}, All {num_of_generations} completions rejected")
                
                except Exception as e:
                    with open(log_file, 'a') as log_f:
                        log_f.write(f"Error in generation: {e}\n")
                        import traceback
                        log_f.write(traceback.format_exc() + "\n")
                    print(f"Error in generation for example {idx} (Line {line_number}): {e}")
                    continue
                
            except Exception as e:
                with open(log_file, 'a') as log_f:
                    log_f.write(f"Error processing example: {e}\n")
                    import traceback
                    log_f.write(traceback.format_exc() + "\n")
                print(f"Error processing example {idx} (Line {line_number}): {e}")
                continue
    
    return accepted_count

def process_file(data_file, image_folder, accu_reward_method, output_dir, args, reward_funcs):
    """Process a single data file with its corresponding image folder"""
    print(f"Processing {data_file} with {image_folder}")
    
    # Create output filename based on input filename
    base_filename = os.path.basename(data_file)
    output_filename = f"processed_{base_filename}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full output file path
    output_file_path = os.path.join(output_dir, output_filename)
    
    # Check for existing processed examples to support proper resuming
    processed_line_numbers = set()
    last_processed_line = -1
    if os.path.exists(output_file_path):
        print(f"Found existing output file {output_file_path}, checking for processed examples...")
        try:
            with open(output_file_path, 'r') as existing_f:
                for line in existing_f:
                    try:
                        item = json.loads(line)
                        if 'input_line_number' in item:
                            line_num = item['input_line_number']
                            processed_line_numbers.add(line_num)
                            last_processed_line = max(last_processed_line, line_num)
                    except json.JSONDecodeError:
                        continue
            print(f"Found {len(processed_line_numbers)} already processed examples")
            print(f"Last processed line number: {last_processed_line}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
    
    # Determine where to start processing (either from args.skip_lines or from last processed line)
    # If args.skip_lines is provided and greater than last_processed_line, use args.skip_lines
    # Otherwise, use last_processed_line + 1 to start from the next unprocessed line
    start_line = max(args.skip_lines, last_processed_line + 1) if args.skip_lines > 0 else (last_processed_line + 1)
    print(f"Starting processing from line {start_line}")
    
    # Collect data for rejection sampling
    image_paths = []
    questions = []
    ground_truths = []
    methods = []
    original_image_paths = []
    line_numbers = []
    
    # Count total lines in the file for progress reporting
    total_lines = get_line_count(data_file)
    print(f"Total lines in {data_file}: {total_lines}")
    
    with open(data_file, 'r') as f:
        for line_idx, line in enumerate(f):
            # Skip lines up to the starting line
            if line_idx < start_line:
                continue
                
            # Skip lines that have already been processed (in case of non-sequential processing)
            if line_idx in processed_line_numbers:
                print(f"Skipping already processed line {line_idx}")
                continue
                
            try:
                item = json.loads(line)
                if 'image' in item:
                    image_paths.append(os.path.join(image_folder, item['image']))
                    questions.append(item['conversations'][0]['value'].replace('<image>', ''))
                    
                    # Handle solution that could be a float or string
                    solution_value = item['conversations'][1]['value']
                    if isinstance(solution_value, str):
                        ground_truths.append(solution_value.replace('<answer>', '').replace('</answer>', '').strip())
                    else:
                        ground_truths.append(str(solution_value))
                    
                    methods.append(item.get('accu_reward_method', accu_reward_method))
                    original_image_paths.append(item['image'])
                    line_numbers.append(line_idx)  # Store the line number for tracking
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON at line {line_idx}, skipping")
                continue
    
    if not image_paths:
        print(f"No new examples to process in {data_file}")
        return 0, last_processed_line + 1
        
    print(f"Found {len(image_paths)} new examples to process")
    
    # Set up generation kwargs
    generation_kwargs = {
        "model": args.generation_model,
        "temperature": args.generation_temperature,
        "max_tokens": args.generation_max_tokens,
        "top_p": args.top_p,
    }
    
    # Run rejection sampling for this file
    accepted_count = rejection_sampling_vqa(
        image_paths=image_paths,
        questions=questions,
        ground_truths=ground_truths,
        output_file_path=output_file_path,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        generation_kwargs=generation_kwargs,
        reward_funcs=reward_funcs,
        accu_reward_methods=methods,
        original_image_paths=original_image_paths,
        line_numbers=line_numbers,
        num_of_generations=args.num_of_generations,
    )
    
    print(f"File {data_file} completed. Accepted {accepted_count} out of {len(questions)} examples.")
    print(f"Processed lines {start_line} to {start_line + len(questions)} out of {total_lines}")
    
    return accepted_count, last_processed_line + len(image_paths)

def main(script_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs.split(":")]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets to get questions and ground truths
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_methods is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_methods.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
    
    # Ensure output directory exists
    output_dir = script_args.output_file_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a resume file to track progress
    resume_file = os.path.join(output_dir, "resume_state.json")
    resume_state = {}
    
    # Load existing resume state if available
    if os.path.exists(resume_file) and script_args.use_resume_file:
        try:
            with open(resume_file, 'r') as f:
                resume_state = json.load(f)
            print(f"Loaded resume state from {resume_file}")
        except Exception as e:
            print(f"Error loading resume state: {e}")
            resume_state = {}
    
    # Process files sequentially instead of in parallel
    total_accepted = 0
    for idx, (data_file, image_folder, accu_reward_method) in enumerate(zip(data_files, image_folders, accu_reward_methods)):
        try:
            # Process the file - the function now determines where to start based on output file
            accepted_count, last_line = process_file(
                data_file, 
                image_folder, 
                accu_reward_method, 
                output_dir, 
                script_args, 
                reward_funcs
            )
            total_accepted += accepted_count
            
            # Update resume state
            file_key = os.path.basename(data_file)
            if file_key not in resume_state:
                resume_state[file_key] = {}
            
            resume_state[file_key]["last_processed_line"] = last_line
            resume_state[file_key]["last_processed_time"] = datetime.now().isoformat()
            
            # Save resume state
            if script_args.use_resume_file:
                with open(resume_file, 'w') as f:
                    json.dump(resume_state, f, indent=2)
            
            print(f"File {data_file} processing completed with {accepted_count} accepted examples")
            print(f"Updated resume state: last processed line = {last_line}")
        except Exception as e:
            print(f"Error processing file {data_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All processing completed. Total accepted examples: {total_accepted}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_paths", type=str, required=True)
    parser.add_argument("--image_folders", type=str, required=True)
    parser.add_argument("--reward_methods", type=str, default=None)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--openai_base_url", type=str, required=True)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_max_tokens", type=int, default=2048)
    parser.add_argument("--generation_model", type=str, required=True)
    parser.add_argument("--reward_funcs", type=str, default="accuracy:format", help="Colon-separated list of reward functions to use")
    parser.add_argument("--skip_lines", type=int, default=0, help="Minimum line number to start processing from. The script will automatically determine the starting point based on already processed examples, but will not go below this value.")
    parser.add_argument("--num_of_generations", type=int, default=1, help="Number of generations to try for each example before rejecting")
    parser.add_argument("--use_resume_file", action="store_true", help="Use resume file to track progress across runs")
    
    args = parser.parse_args()
    main(args)
