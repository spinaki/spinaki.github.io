---
title: "The Illustrated Direct Preference Optimization (DPO)"
date: 2025-10-20
categories:
  - Deep Learning
  - LLMs
tags:
  - RLHF
  - DPO
  - Math
share: false
related: false
toc: true
toc_label: "Contents"
toc_icon: "cog"
excerpt: "Deriving DPO from scratch: How we moved from complex Reinforcement Learning to simple Classification for aligning LLMs."
---

If you have trained a Large Language Model (LLM), you know the drill: Pre-training teaches the model to speak; Supervised Fine-Tuning (SFT) teaches it to follow instructions. But how do we teach it to be *good*? How do we align it with vague human values like "helpfulness" or "safety"?

For a long time, the answer was **RLHF (Reinforcement Learning from Human Feedback)**. It involved a complex pipeline of training a separate Reward Model and then using PPO (Proximal Policy Optimization) to update the LLM. It was unstable, difficult, and computationally expensive.

Then came **DPO (Direct Preference Optimization)** in 2023. It showed us that we can skip the Reward Model and the Reinforcement Learning entirely. We can optimize for human preferences using a simple classification loss.

In this post, we will derive DPO from scratch, starting from the "Magic" analytical solution of the RLHF objective, and visualize exactly how it works.

## 1. The Context: The Preference Problem

Language models are trained to predict the next token (Cross-Entropy). This makes them great mimics, but bad judges.
* **SFT (Supervised Fine-Tuning):** We show the model: "When I ask X, say Y."
* **The Problem:** Humans are better at *recognizing* a good answer than writing one. It is easier to say "Response A is better than Response B" than to write the perfect Response A from scratch.

This gives us a dataset of **Pairs**:
* **Prompt ($x$):** "Write a poem about rust."
* **Winner ($y_w$):** A creative, melancholic poem.
* **Loser ($y_l$):** A boring, technical description of oxidation.

We want a model that maximizes the probability of generating the **Winner**.

### The Bradley-Terry Model

To do this mathematically, we assume there is a hidden "Reward Score" $r(x, y)$ for every response. The **Bradley-Terry (BT) model** tells us the probability that one item beats another based on their score difference:

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

Where $\sigma$ is the sigmoid function.
* If $r(Winner) \gg r(Loser)$, the probability approaches 1.
* If $r(Winner) \approx r(Loser)$, the probability is 0.5 (a toss-up).

In standard RLHF, we train a neural network to estimate this $r(x, y)$. **DPO skips this.**

---

## 2. The "Magic" Analytical Solution

Before we get to DPO, we must look at the objective of RLHF. We want a policy $\pi$ that maximizes reward but does not drift too far from our original reference model $\pi_{ref}$ (to prevent the model from outputting gibberish just to game the reward).

$$\max_{\pi} \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x, y)] - \beta \mathbb{D}_{KL}(\pi(y|x) || \pi_{ref}(y|x))$$

Usually, we solve this using PPO (gradient ascent). But it turns out, this specific equation has a **closed-form analytical solution**. We can solve it with pen and paper without training a neural network yet.

### Step-by-Step Derivation

Let's expand the KL divergence term:

$$\max_{\pi} \sum_{y} \pi(y|x) r(x, y) - \beta \sum_{y} \pi(y|x) \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$$

We can rewrite the reward $r(x, y)$ as a logarithm: $r(x, y) = \beta \log \left( \exp \left( \frac{r(x, y)}{\beta} \right) \right)$. Substituting this back:

$$\max_{\pi} \sum_{y} \pi(y|x) \left[ \beta \log \left( \exp \left( \frac{r(x, y)}{\beta} \right) \right) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

Combine the logs:

$$\max_{\pi} \beta \sum_{y} \pi(y|x) \log \left( \frac{\pi_{ref}(y|x) \exp(\frac{r(x, y)}{\beta})}{\pi(y|x)} \right)$$

Now, let's introduce a normalizing constant (partition function) $Z(x) = \sum_{y} \pi_{ref}(y|x) \exp(\frac{r(x, y)}{\beta})$. We multiply and divide by $Z(x)$ inside the log:

$$\max_{\pi} \beta \sum_{y} \pi(y|x) \log \left( \frac{Z(x) \cdot \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{r(x, y)}{\beta})}{\pi(y|x)} \right)$$

Separating terms:

$$\max_{\pi} \beta \left[ \underbrace{\sum \pi(y|x) \log Z(x)}_{\log Z(x)} - \underbrace{\sum \pi(y|x) \log \frac{\pi(y|x)}{\frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{r(x, y)}{\beta})}}_{\text{KL Divergence}} \right]$$

This leaves us with:

$$\max_{\pi} \beta \left[ \log Z(x) - \text{KL}(\pi || \pi^*) \right]$$

where we define the **Optimal Policy $\pi^*$** as:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

To maximize the equation, we simply minimize the KL divergence to 0. This implies $\pi = \pi^*$.

**The Insight:** The optimal policy is just the reference model scaled by the exponential reward.

---

## 3. The DPO Inversion

This is where DPO changes the game. We have a formula relating the **Optimal Policy** ($\pi^*$), the **Reference** ($\pi_{ref}$), and the **Reward** ($r$).

Usually, we know $r$ (from a reward model) and try to find $\pi^*$.
DPO asks: **What if we solve for $r$ instead?**

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

Take the log of both sides:

$$\log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} = \frac{r(x, y)}{\beta} - \log Z(x)$$

Rearrange to isolate $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

We have just defined the reward **in terms of the policy itself**. We do not need a separate reward model anymore.

### Plugging into Bradley-Terry

Recall the Bradley-Terry preference model: $P(Winner > Loser) = \sigma(r(Winner) - r(Loser))$.

Let's plug our new definition of $r$ into this.

$$r(y_w) - r(y_l) = \left[ \beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)} + \beta \log Z(x) \right]$$

The partition function $Z(x)$ cancels out. We are left with:

$$r(y_w) - r(y_l) = \beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)}$$

We now just minimize the negative log-likelihood of this probability. This is the **DPO Loss**.

---

## 4. Visualizing the Algorithm

How does this actually look during training? It is surprisingly simple.

<div class="mermaid">
graph TD
    Data[Dataset Pair] -->|Prompt x| Policy[Policy Model &pi;]
    Data -->|Prompt x| Ref[Reference Model &pi;_ref]

    subgraph "Forward Pass (No Generation Needed!)"
    Policy -->|Forward| LogP_W_Pol[LogProb Winner]
    Policy -->|Forward| LogP_L_Pol[LogProb Loser]
    Ref -->|Forward| LogP_W_Ref[LogProb Winner]
    Ref -->|Forward| LogP_L_Ref[LogProb Loser]
    end

    subgraph "Implicit Reward Calculation"
    LogP_W_Pol & LogP_W_Ref --> Ratio_W["log( Pol(W) / Ref(W) )"]
    LogP_L_Pol & LogP_L_Ref --> Ratio_L["log( Pol(L) / Ref(L) )"]
    end

    Ratio_W & Ratio_L --> Diff["Difference * Beta"]
    Diff --> Loss["Binary Cross Entropy Loss"]
    Loss --> Update["Update Policy Weights"]

    style Policy fill:#f9f,stroke:#333,stroke-width:2px
    style Data fill:#ccf,stroke:#333,stroke-width:2px
    style Loss fill:#f99,stroke:#333,stroke-width:2px
</div>

### An Illustrated Example

Imagine we have a prompt: **"What is 2+2?"**

1.  **Winner ($y_w$):** "4"
2.  **Loser ($y_l$):** "5"

We run these through our **Reference Model** (frozen) and our **Current Policy** (training).

| Model | Log-Prob of "4" (Winner) | Log-Prob of "5" (Loser) |
| :--- | :--- | :--- |
| **Reference** | -1.0 (36%) | -2.0 (13%) |
| **Policy (Start)** | -1.0 (36%) | -2.0 (13%) |

At the start, the Policy equals the Reference.
* **Implicit Reward (Winner):** $\log(-1.0) - \log(-1.0) = 0$
* **Implicit Reward (Loser):** $\log(-2.0) - \log(-2.0) = 0$
* The model has no preference yet.

**The Optimization Step:**
DPO wants to make the **Implicit Reward Difference** positive.
It wants: $\frac{\pi(Winner)}{\pi_{ref}(Winner)} > \frac{\pi(Loser)}{\pi_{ref}(Loser)}$

So the gradients will push the weights to:
1.  **Increase** Policy Log-Prob of "4" (e.g., to -0.8).
2.  **Decrease** Policy Log-Prob of "5" (e.g., to -2.5).

**After one step:**

| Model | Log-Prob of "4" (Winner) | Log-Prob of "5" (Loser) | Ratio (Implicit Reward) |
| :--- | :--- | :--- | :--- |
| **Reference** | -1.0 | -2.0 | N/A |
| **Policy** | **-0.8** (Higher) | **-2.5** (Lower) | **Winner > Loser** |

The Policy is now "more likely" to say 4 than the Reference was, and "less likely" to say 5 than the Reference was. That relative shift **is** the reward.

---

## 5. Intuitions and Takeaways

### The "Implicit" Judge

In PPO, you have a separate Teacher (Reward Model) grading the Student (Policy).
In DPO, the **Student becomes the Judge**.
By relating the reward directly to the optimal policy, DPO enforces that the model's own probability distribution *must* reflect the preferences. If the model knows "A is better than B", it *must* assign a higher probability to A (relative to the reference).

### Stability via "Weighted Push-Pull"

The gradient of DPO reveals a beautiful property. It pushes the Winner up and the Loser down, but it is weighted by $\sigma(\hat{r}_{loser} - \hat{r}_{winner})$.

* **If the model is wrong** (thinks Loser > Winner): The weight is high. The model learns aggressively.
* **If the model is right** (thinks Winner > Loser): The weight approaches 0. The model stops changing.

This prevents the "falling off a cliff" instability seen in PPO, where models often collapse if updated too aggressively on data they already know.

### Analogy: Clearing the Fog

* **PPO** is like navigating a foggy mountain. You take a step, check your GPS (Reward Model), and adjust. You might walk off a cliff.
* **DPO** clears the fog. The math derivation gives us the exact coordinates of the summit. We just use supervised learning to teleport the model directly to those coordinates.

---

## References & Further Reading

* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
* [Stanford CS336 Lecture Notes on RLHF](https://stanford-cs336.github.io/)
* [Hugging Face TRL Library](https://huggingface.co/docs/trl/index)
