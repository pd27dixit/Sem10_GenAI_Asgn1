20CS30069

SPECIFIC PYTHON VERSION
Python version: 3.10.12


SPECIFIC LIBRARY REQUIREMENTS

Essential Libraries:
torch (PyTorch) – for deep learning computations
torchvision (if needed) – for additional PyTorch utilities
torch.nn – for defining neural networks
torch.optim – for optimization algorithms

Hugging Face Transformers & Related Libraries:
transformers – for pre-trained NLP models like BERT and GPT
datasets – for handling large-scale NLP datasets
trl==0.11 – for reinforcement learning with transformers
accelerate – for optimizing training speed

Performance & Optimization:
torch.cuda.amp – for mixed-precision training
torch.nn.functional – for additional neural network operations

Metrics & Evaluation Libraries:
nltk – for BLEU score evaluation
rouge_score – for ROUGE score evaluation
evaluate – for NLP model evaluation
Progress Tracking & Logging:
tqdm – for progress bars
wandb – for experiment tracking

To install a library do this in python notebook - '!pip install <library_name>'
For a particular version '!pip install <library_name>==<version>'


a) DETAILS ON IMPLEMENTATION OF REWARD MODEL
The goal of the reward model is to learn a scoring mechanism that differentiates between preferred and less preferred samples based on textual inputs. This is achieved using a BERT-based neural network and trained using the Bradley-Terry loss function, which is commonly used in preference-based learning.

  1. Model Architecture
    - Built using BERT (bert-base-uncased) as the feature extractor.
    - A linear layer (score_head) added on top of BERT’s pooled output -> generate reward score.
    - The BERT embeddings are fine-tuned during training to learn task-specific features.
  
  2. Loss function
    - The Bradley-Terry loss is used to ensure that a more preferred sample is scored higher than a less preferred one.
        L = -(1/N) * (summation till N) [log (sigmoid  (s_more - s_less))]
        where s_more and s_less are scores for the more and less preferred inputs
  3. Data Processing
    - Data is loaded from a CSV file into a Hugging Face Dataset.
    - Each sample contains:
        more_prefered_input_ids and more_prefered_attention_mask for the preferred text.
        less_prefered_input_ids and less_prefered_attention_mask for the less preferred text.
    - Tokenization is done using BERT tokenizer.

  4. Training Setup
    - PyTorch Accelerator is used for efficient training on available hardware (CPU/GPU).
    - AdamW optimizer is used with a learning rate of 5e-5.

  5. Training Loop
    - Each batch contains:
        Forward pass through RewardModel for both preferred and less preferred samples.
        Bradley-Terry loss computation.
        Backpropagation and optimization step.

        


b) DETAILS ON IMPLEMENTATION OF RLHF (PPO) MODEL
Reinforcement Learning with Human Feedback is used to fine-tune language models by aligning their behavior with human preferences. Here, we use Proximal Policy Optimization (PPO) to optimize a GPT-2 Medium model based on a previously trained reward model.

  1. Model and Tokenizer Setup
    - GPT-2 Medium as the base language model and the reference model (frozen this)
    - tokenizer is initialized using BERT (since reward model was trained on BERT, so consistency of tokens when training)
    - to support PPO training, base model is loaded using AutoModelForCausalLMWithValueHead
  
  2. Cloning and Freezing the Reference Model
    - A reference model is created from the base model.
    - It is frozen (requires_grad = False), remains unchanged during training, acting as a baseline for KL divengence

  3. PPO Configuration

    - A PPOConfig object is defined with the following key parameters:
        init_kl_coef=0.2: Ensures stability, control the KL divergence penalty.
        target_kl=0.05: Prevent drastic policy updates.
        learning_rate=2e-6: fine-tuning by small penalty .
        batch_size=16, mini_batch_size=8: Batch sizes are set.

  4. Loading and Preprocessing Data
    - Training dataset (preference_train.csv)loaded, with columns 'Question' and 'More_Prefered'.

    - It is converted into a Hugging Face dataset format where each sample contains:
        query: The question or prompt.
        response: The more preferred response from the dataset.

  5. Training Setup
    - Gradient accumulation, stabilizes training (gradient_accumulation_steps=4).
    - The optimizer AdamW with a linear scheduler that has 500 warmup steps.

  6. Training Loop with PPO
    - The training process follows these steps:
        Tokenization: Queries tokenized with padding and attention masks.
        Response Generation:Model generates responses using nucleus sampling and temperature scaling (temperature=0.7).
        Reference Response Tokenization: Preferred responses from the dataset are tokenized.
        Reward Calculation:
          - Our pre-trained reward model ( Step a) ) scores both the generated and reference responses.
          - Reward scores are clamped between -1 and 1.

        PPO Optimization:
          - The PPO step updates the policy model using:
              input_ids (queries)
              generated_input_ids (model responses)
              reward_scores (from the reward model)

          - Optimization steps are performed, and learning rates are adjusted using the scheduler.

  7. Evaluation and Logging
    - The total reward is accumulated, and the average reward is computed per epoch.
    - Debugging information such as log probabilities monitors training.


  8. Special considerations
    - max_new_tokens=20 ensures that generated responses do not become too long.
    - Nucleus sampling (top_p=0.95) improves diversity while maintaining coherence.
    - Ensuring that rewards are computed on the CPU (reward_model(more_input_ids.cpu())) prevents memory overload.
    - Regular calls to torch.cuda.empty_cache() prevent excessive GPU memory consumption.

c) DETAILS ON IMPLEMENTATION OF DPO MODEL
Direct Preference Optimization (DPO) used for fine-tuning a language model based on human preferences. The goal is to train a model to align its responses with preferred outputs over less-preferred ones without requiring explicit reward modeling.

  1. Model and Tokenizer Setup
    - GPT-2 Medium model (openai-community/gpt2-medium) is used as the base model.
    - Reference model (identical to the base model) is instantiated and frozen to serve as a comparison during training.
    - AutoTokenizer init  and configured with an end-of-sequence (EOS) token as the padding token.

  2. Dataset Preparation
    - Dataset containing prompts, more preferred responses, and less preferred responses
    - A custom PyTorch Dataset class (PreferenceDataset) facilitate data loading.
    - The DataLoader batches the dataset for training, allowing shuffling for randomness.

  3. Tokenization Strategy
    - A tokenize_batch function, to tokenize prompt-response pairs while ensuring padding and truncation to a maximum sequence length (MAX_SEQ_LEN = 512).
    - The tokenized inputs are moved to the CUDA device (if available) for efficient computation.

  4. Direct Preference Optimization (DPO) Loss Function
    - The dpo_loss function computes the log probability differences between the base model and reference model for more-preferred and less-preferred responses.
    - Logits are extracted from the last token position (logits[:, -1, :]) for both more preferred and less preferred responses.
    - The log-softmax function is applied to obtain normalized probabilities.

  5. Training Strategy

    - (GRAD_ACCUM_STEPS = 4) manage memory usage -> accumulating gradients over multiple steps before updating weights.

    - Automatic Mixed Precision (AMP) is enabled to speed up training and reduce memory consumption.

    - Gradient checkpointing is activated to save memory by recomputing intermediate activations instead of storing them.

    - AdamW Optimizer is used with a learning rate of 1e-5.

    - Memory Optimization includes torch.cuda.empty_cache() and gc.collect() to free unused memory.

  6. Training Loop

    - The training loop iterates through epochs (EPOCHS = 1) and batches from the DataLoader.

    - For each batch:
        The dpo_loss is computed.
        The loss is backpropagated using AMP scaling.
        Weights are updated every 4 steps to reduce frequent updates and improve stability.
        GPU memory is freed periodically to avoid memory leaks.

    - After each epoch, the average loss is printed for tracking progress.


EXTRA POINTERS:
I have trained on entire datasets (both for RLHF and DPO) but BLEU Score still around 0.056 only.
Some might have trained on 10% dataset only, and might get good scores (but it means that they were not able to load or use GPU for entire training)

I have done my code in Kaggle notebook
hence input paths looks like this /kaggle/input/folder1/file

a) for Reward model training 
  - it took around 2 hrs to train on entire dataset
  - I stored model in output directory 
      torch.save(reward_model.state_dict(), "reward_model.pth")
  - But since sessions get over after sometimes, and to avoid the model getting forgotten, I did this.
  - I downloaded the model, and put it in this directory /kaggle/input/rewardm/reward_model.pth
    and also in /kaggle/input/more-uploads/reward_model.pth
  - Did this to ease out loading later

b) for PPO RLHF Training
  - it took around 7.5 hrs to train entire dataset
  - in notebook it has cut short the output logs upto 53% only, though full training got completed
  - again for notebook not to forget my saved model, I did the following
  - I downloaded and put in directory /kaggle/input/saved-models/ppo_fine_tuned_model

c) for DPO Training
  - it took around 2.5 hrs to train on entire dataset
  - again for notebook not to forget my saved model, I did the following
  - I downloaded and put in directory /kaggle/input/saved-models/gpt2-medium-dpo

d) Evalution 
  - it took 46 min each for DPO and PPO models 


I have in middle, always emptied cache and release GPU memory - encountered lot of OutOfMemory Error for CUDA
To match the naming convention of assignment, in the end I have renamed all model and files.
I have also coded to make downloaded links, I was unable to download from kaggle.

