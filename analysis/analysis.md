# Analysis

## Table of Contents
- [Notation](#notation)
    - [Question Numbers](#question-numbers)
    - [Acronyms](#acronyms)
- [Key Findings](#key-findings)
- [Overall Results](#overall-results)
    - [Summary Comparison](#summary-comparison)
        - [Initial Analysis](#initial-analysis)
- [Error Taxonomy and Distribution](#error-taxonomy-and-distribution)
    - [Taxonomy](#taxonomy)
        - [Extraction Failure](#extraction-failure)
        - [Math Error](#math-error)
        - [Misunderstood Question](#misunderstood-question)
        - [Logical Reasoning Error](#logical-reasoning-error)
        - [Hallucination](#hallucination)
        - [Forgetting](#forgetting)
        - [Unknown](#unknown)
    - [Distribution](#distribution)
- [Cross-Strategy Comparison](#cross-strategy-comparison)
    - [Per-Question Breakdown](#per-question-breakdown)
    - [Outcome Distribution Overall](#outcome-distribution-overall)
    - [Outcome Distribution by Error Type](#outcome-distribution-by-error-type)
        - [Chain of Thought](#chain-of-thought)
        - [Baseline](#baseline)
    - [Comparison Analysis](#comparison-analysis)
        - [1. Error type persists across strategies](#1-error-type-persists-across-strategies)
        - [2. CoT is great at math and (possibly) at not hallucinating](#2-cot-is-great-at-math-and-possibly-at-not-hallucinating)
- [Refinement Attempts](#refinement-attempts)
    - [One Shot and Chain of Thought](#one-shot-and-chain-of-thought)
        - [OS-and-CoT Results](#os-and-cot-results)
        - [1. Why no apparent CoT influence?](#1-why-no-apparent-cot-influence)
        - [2. Why is the ESR so good?](#2-why-is-the-esr-so-good)
    - [One Shot of Chain of Thought](#one-shot-of-chain-of-thought)
        - [OS-of-CoT Results](#os-of-cot-results)
        - [1. Why is the ESR so bad?](#1-why-is-the-esr-so-bad)
        - [2. Why does OS-of-CoT have a slightly worse AoES than CoT does?](#2-why-does-os-of-cot-have-a-slightly-worse-aoes-than-cot-does)
- [Tradeoffs](#tradeoffs)
- [Limitations and Future Direction](#limitations-and-future-direction)
    - [Depth of Existing Analysis](#depth-of-existing-analysis)
    - [The Uncertainty of Taxonomy](#the-uncertainty-of-taxonomy)
        - [The ambiguity of the true source of an error](#the-ambiguity-of-the-true-source-of-an-error)
        - [The presence of multiple errors](#the-presence-of-multiple-errors)
    - [Tradeoff Resolution](#tradeoff-resolution)
    - [Prompt Adjustments](#prompt-adjustments)
        - [Wholly New Ideas](#wholly-new-ideas)
        - [Formally Testing Previous Informal Tests](#formally-testing-previous-informal-tests)
    - [Conclusion](#conclusion)

## Notation

### Question Numbers

Whenever reference is made to a question number, it is by its index in the test split of GSM8K, which corresponds to the `index` value for each result record `json` in one of the `jsonl` output files, including in the first 50 errors file mentioned below in the [Error Taxonomy and Distribution](#error-taxonomy-and-distribution) introduction.

### Acronyms

| Acronym | Definition |
|:---|---:|
| AO | Answer Only |
| BL | Baseline |
| CoT | Chain of Thought |
| OS | One Shot |
| OS-and-CoT | One Shot and Chain of Thought |
| OS-of-CoT | One Shot of Chain of Thought |
| AoES | Accuracy on Extraction Success |
| ESR | Extraction Success Rate |

## Key Findings

1. **Concrete examples and abstract instruction in prompts compete for signal, and concrete examples win.** Combining OS and CoT to make OS-and-CoT caused the model to imitate the more concise style of the OS example without retaining CoT's AoES advantage (71.2% to 57.6%) and to even improve ESR beyond OS's (90.7% to 97.0%).  
    See: [Why no apparent CoT influence?](#1-why-no-apparent-cot-influence)

2. **There is a tension between ESR and AoES where the lever is the extent to which the model thinks step-by-step.** AO and BL at one extreme do not have any thinking in the response, and thus possess strong ESR (97.4%, 92.7%) coupled with abysmal AoES (9.2%, 8.2%). CoT and OS-of-CoT are on the opposite end with poor ESR and excellent AoES (51.9%, 38.7% ESR and 71.2%, 68.4% AoES).  
    See: [Tradeoffs](#tradeoffs)

3. **Error types persist across prompting strategies.** 8/10 CoT errors on questions where OS had a logical reasoning error were also logical reasoning errors for CoT, and the same persistence occurs in 5/9 misunderstanding the question errors and 2/2 forgetting errors. This suggests model-level rather than prompt-level limitations.  
    See: [Error type persists across strategies](#1-error-type-persists-across-strategies)

4. **CoT disproportionately corrects OS math errors compared to other error types.** It corrected 5/6 math errors vs. 5/15 for logical reasoning, 4/13 for misunderstanding, and 0/2 for forgetting. Hallucinations may be improved, with 2/3 corrected.  
    See: [CoT is great at math and (possibly) at not hallucinating](#2-cot-is-great-at-math-and-possibly-at-not-hallucinating)

5. **OS-and-CoT is the strongest strategy.** Its 55.9% accuracy is carried by an excellent ESR (97%) and supported by an acceptable AoES (57.6%).  
    See: [Summary Comparison](#summary-comparison) and [One Shot and Chain of Thought](#one-shot-and-chain-of-thought)

## Overall Results

### Summary Comparison

This summary table is replicated here from the [README](../README.md#summary-comparison-between-prompting-strategies) for the reader's convenience.

| Strategy | Accuracy | Extraction Success Rate  | Accuracy on Extraction Success |
|:---|---:|---:|---:|
| **One Shot and Chain of Thought** | 55.9% | 97.0% | 57.6% |
| **One Shot** | 51.9% | 90.7% | 57.2% |
| **Chain of Thought** | 37.0% | 51.9% | 71.2% |
| **One Shot of Chain of Thought** | 26.5% | 38.7% | 68.4% |
| **Answer Only** | 8.9% | 97.4% | 9.2% |
| **Baseline** | 7.6% | 92.7% | 8.2% |

#### Initial Analysis

OS-and-CoT has the overall best accuracy due to its excellent ESR and good AoES, with OS not far behind. See the [Refinement Attempts](#refinement-attempts) section below for extensive commentary on OS-and-CoT as well as OS-of-CoT. Each of these were added as a result of error analysis.

CoT has by far the best AoES, but remains middling in overall accuracy due to a poor ESR, at which it is only better than OS-of-CoT.

Merely suggesting thinking step by step (CoT) causes a drastic drop in ESR compared to BL (51.9% vs. 92.7%). This goes to show that it is easy to overload smaller models on instructions—add the "Explain your answer step by step," and the "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'" is often forgotten.

I suspect that the above does not occur for OS because while BL and CoT have a formatting-specific one-shot embedded in the prompt ("For example: '#### 42'"), OS's example response includes appropriate formatting as well, which makes it two-shot with respect to formatting, strengthening the model's ESR.

BL is the worst, even worse than AO, even when ignoring extraction as a factor. This is surprising since using its response to think through an answer typically makes a model perform better—that is the whole idea behind CoT as a methodology, and AO makes it answer without any of that thinking. However, as the [Cross-Strategy Comparison](#cross-strategy-comparison) section below reveals, BL rarely contained a response beyond the answer, either. But that only explains why BL does not do meaningfully better than AO, not why it does meaningfully worse, so it remains surprising.

## Error Taxonomy and Distribution

Before having added OS-of-CoT and OS-and-CoT, of which I had not yet conceived, I wanted to analyze the errors of the existing prompting methods with the hope that doing so would inform how to refine the prompts to create newer, better ones. Of course, that is what led to OS-of-CoT and OS-and-CoT.

Since OS was the best performer among the original four in terms of overall accuracy, I chose to analyze its results to start.

I collected the first 50 errors from the OS results, using a script I wrote to assist (see [find_errors.py](scripts/find_errors.py) in the scripts directory), which parses a results file for a given number of regular errors / extraction failures and writes them to a new `jsonl` file. **A copy of the first 50 errors file can be found [here](2026-05-08_13-34-48_qc100_one-shot_first_50_errors.jsonl) in the analysis directory.** This may be of use for reference when specific problems are referred to throughout this document. Consider using a jsonl reader website like https://jsonl.co/ to make the file more readable. Note that although the errors were drawn from a 100 question run rather than the full 1319, the responses are identical to the full run's first 100 because I used the same configuration, and greedy decoding ensures consistent responses for the same configuration.

I then manually categorized the 50 errors into the following categories. I initially had in mind the first four. The final three were additions I made during the process.

### Taxonomy 
1. Extraction Failure 
2. Math Error
3. Misunderstood Question
4. Logical Reasoning Error
5. Hallucination
6. Forgetting
7. Unknown

#### Extraction Failure

Extraction failures are already discussed in the README and are very visible given that they are easy to recognize: if extracting the numerical answer from the model's full response does not work, then it is an extraction failure. Thus, they are in the results and summary files and do not need to be manually categorized like the other error types. For AO, an extraction failure occurs when the model responds with anything other than a string of a float, possibly with commas. Otherwise, this usually means the model does not include the `#### `&nbsp;before the answer, but can also mean that it includes some information after the `####` but before the number, like `#### Final Answer: 42`. Information after the number, separated by a space, does work, however, as in `#### 42 melons`.

#### Math Error

A math error is the most straightforward to understand and to see in the model's response: the model sets up the right math, but then executes upon it incorrectly. For example, in problem 67, the model is adding some numbers, and is off by 100: "Therefore, there are 175+140+280 = <<175+140+280=695>>695 gems in total."

#### Misunderstood Question

In this case, the model takes a specific fact stated in the problem and gets it wrong or misses it entirely. This category is best explained by example, so consider the following several:
- Six years older becomes six years younger (21)
- Calories per serving becomes per bag (43)
- There are as many rabbits as cats and dogs minus 12 becomes as many rabbits as just cats plus dogs (94)

#### Logical Reasoning Error

In logical reasoning errors, the model does the wrong things with the right facts. Logical reasoning errors necessarily span the widest range of ways of being wrong, for there are many different ways to embody faulty logic. Here are two examples:
- It might set up the wrong mathematical relationship, subtracting instead of adding, as in problem 41, where it concludes that a safe range to attack the dragon (which has an 1000 foot range with flames) with the gold javelin and sapphire gemstone that through their combined power have a 1200 foot is... -200 feet.
- Or it just loses the connections between the pieces in a more abstract way, as in problem 46, where post-its are being purchased and used, and none of the numbers are considered appropriately (the amount she starts with is immediately considered used and the amount she starts with plus the amount remaining at the end are inexplicably subtracted from the amount she is said to use to determine how many... she used).

#### Hallucination

Hallucinations are when the model introduces information that is nowhere in the problem. If it misinterprets, misapplies, or fails to apply information that is in the problem, that is one of the other categories. There is a small sample size for hallucinations, with only 3 in these 50, but in all of the cases, the model introduced a number not present in the prompt. It pulls out of thin air, for example, the number of blue ties purchased, when that information is not given (47).

#### Forgetting

Here, a model acknowledges information early, and then forgets to account for it later. Unlike a logical reasoning error, it does not do the wrong thing with the information; rather, it does nothing with it. And unlike misunderstanding the question, it explicitly mentions the forgotten information. In question 50, it acknowledges that we know about a farm's egg production per day, and that we need to find the egg farmer's profit per week, but it fails to do the conversion from day to week.

#### Unknown

In some cases, it is not clear which category an error fits into. The only source of information is the model's response, and sometimes that is insufficient to categorize with confidence. For OS, there was only one case of this in the 50 problems I examined, question 77. In this case, the model does not give enough information for it to be clear whether it misread a fact or it misapplied logic to correctly-read facts; in other words, it is not clear whether it is an issue of misunderstanding the question or logical reasoning. To make the point clearer, AO responses, assuming the model actually does give only the answer, are always categorized as unknown errors when they are wrong—there is not enough reasoning visible to make a reliable conclusion as to what type of error it is. While other prompting strategies do usually have reasoning, that does not mean there is enough information in their responses, either. See more on this in the [The ambiguity of the true source of an error](#the-ambiguity-of-the-true-source-of-an-error) subsection of [Limitations and Future Direction](#limitations-and-future-direction) below.

### Distribution

| Error type | Count | Questions |
|---|---:|---|
| Logical Reasoning Error | 15 | 0, 2, 5, 7, 15, 19, 20, 37, 41, 46, 57, 63, 64, 76, 98 |
| Misunderstood Question | 13 | 11, 21, 29, 43, 44, 58, 60, 62, 65, 66, 87, 94, 97 |
| Extraction Failure | 10 | 3, 8, 14, 30, 33, 40, 51, 55, 79, 92 |
| Math Error | 6 | 12, 31, 45, 67, 73, 82 |
| Hallucination | 3 | 36, 47, 85 |
| Forgetting | 2 | 50, 54 |
| Unknown | 1 | 77 |
| **Total** | **50** | |

Logical reasoning errors are the most preponderant, indicating that reducing their frequency would be most effective in improving performance of OS. However, it is vague what connects one logical reasoning error to another causationally. This category contains a set of errors for which the base issue could be caused in a wide number of ways. So I am skeptical that any type of prompt adjustment would help them universally.

Misunderstanding the question is not far behind reasoning in count—close enough that it is certainly possible that in the full dataset, it is actually a more frequent type than logical reasoning errors. And I can readily imagine various ways to possible improvement here. For example, the model could be told to restate the problem or even just to ensure that it understands the question, or similar such additions. These would likely make the model pay more attention to the question which I suspect would improve performance.

There is a drop-off before the frequency of math problems, but there are still enough of these that improving them would be useful. In contrast, hallucination and forgetting are both uncommon enough that trying to improve them is not worthwhile for now.

Improving ESR would certainly be helpful. After all, that is why OS has a better performance than CoT—its ESR is substantially better, enough to overcome its lower AoES. However, resolving an extraction error does not mean that the model will get the right answer. There are many extraction failures where what was failed to be extracted was itself a wrong answer, such as in question 8. Furthermore, I have already worked on improving the ESR informally as mentioned in the [Prompt Formatting](../README.md#prompt-format) section of the README.

Less obviously, the point above regarding fixing extraction not necessarily leading to a correct answer is also true for error types other than extraction failures—that is to say, sometimes, there are multiple other error types in one response (though with a lesser frequency than when one is an extraction failure). For example, question 60 includes both misunderstanding the question and then later a math error. Resolving just one half of that would still produce a wrong answer.

**A note on the sample size:** Compared to the overall 1319 set of test questions in GSM8K, 50 is a small sample size. However, one indication of good representativeness is that the extraction failure rate is very close to that of the rate in the full set. 10/50 or 20% of the failures in this sample are extraction failures compared to 123/635 in the full set or 19.4%. Of course, this does not necessarily mean that the proportions of the other error types are equally accurate, but it is nonetheless a good sign for the accuracy of the distribution in this sample relative to the full set. Crucially, though, this definitely does not mean that the sample size for each error type is necessarily large enough to make meaningful conclusions on it. When considered, for example, "How do forgetting errors tend to look?", it cannot be reliably determined from this data, since we only have a sample size of 2 for them. See more on this in the [Depth of Existing Analysis](#depth-of-existing-analysis) subsection of the [Limitations and Future Direction](#limitations-and-future-direction) section.

## Cross-Strategy Comparison

All of the above discussion in the [Distribution](#distribution) section is just based on the OS data. While useful for determining where the model goes wrong and where improvements would be particularly valuable, that data is insufficient to indicate *how* to improve. Thus, the next course of action I took was to compare the results for these questions to the results of the same questions using different prompting strategies to get more data. Regardless of how similar or different the distribution of error types is, that data will provide empirical evidence for how to improve results beyond the mere intuitive theory in the previous stage.

CoT did better assuming successful extraction, so I thought much could be learned from it. BL was also included in the comparison for completeness. AO was not included because since the model is explicitly instructed to only include the answer, there is no reasoning to evaluate, so it is useless when considering failure types (excluding extraction failures). As you will see in the tables below, BL, in fact, almost never provides reasoning either, so it is mostly useless in this respect, as well. Consequently, this is essentially a comparison between OS and CoT.

OS extraction failures (10) and unknown failures (1) are excluded in the cross-strategy comparison. Extraction behavior is already captured by the automated summary, so manual comparison adds nothing. And improvement on the unknown error is not informative. This leaves 39 questions to compare across prompting strategies.

When compared strategies have an extraction failure, it is acknowledged with the EF flag mentioned in the legend below, but it is ignored for comparison purposes and I still look at the underlying reasoning; something can be learned from considering the model's response regardless of extraction success, since getting the correct answer and formatting correctly for extraction are separate matters.

The following table is sorted by OS error type and then by question number.

### Per-Question Breakdown

#### Legend
 
- `✓` — correct
- `(EF)` — extraction failure flag
- `✓*` — correct answer but scored wrong for a reason that is **not** an EF; see footnote

| Q# | One Shot | Chain of Thought | Baseline |
|---:|---|---|---|
| 0 | reasoning | ✓ | unknown |
| 2 | reasoning | reasoning (EF) | unknown |
| 5 | reasoning | reasoning | unknown |
| 7 | reasoning | reasoning (EF) | ✓ |
| 15 | reasoning | ✓ (EF) | ✓ (EF) |
| 19 | reasoning | ✓ | unknown |
| 20 | reasoning | reasoning | unknown |
| 37 | reasoning | reasoning | unknown |
| 41 | reasoning | misunderstood | unknown |
| 46 | reasoning | reasoning (EF) | unknown |
| 57 | reasoning | ✓ | ✓ (EF) |
| 63 | reasoning | ✓ (EF) | unknown |
| 64 | reasoning | reasoning | unknown |
| 76 | reasoning | forgetting | unknown |
| 98 | reasoning | reasoning (EF) | unknown |
| 11 | misunderstood | math (EF) | math |
| 21 | misunderstood | misunderstood | unknown |
| 29 | misunderstood | ✓ | unknown |
| 43 | misunderstood | reasoning | unknown |
| 44 | misunderstood | misunderstood | unknown |
| 58 | misunderstood | reasoning (EF) | unknown |
| 60 | misunderstood | misunderstood | unknown |
| 62 | misunderstood | misunderstood (EF) | ✓ |
| 65 | misunderstood | ✓ (EF) | unknown |
| 66 | misunderstood | forgetting | unknown |
| 87 | misunderstood | misunderstood (EF) | misunderstood |
| 94 | misunderstood | ✓ (EF) | ✓* |
| 97 | misunderstood | ✓ | unknown |
| 12 | math | reasoning | reasoning |
| 31 | math | ✓ | unknown |
| 45 | math | ✓ (EF) | ✓* |
| 67 | math | ✓ (EF) | unknown |
| 73 | math | ✓ (EF) | unknown |
| 82 | math | ✓ | unknown |
| 36 | hallucination | ✓ | unknown |
| 47 | hallucination | ✓ | unknown |
| 85 | hallucination | misunderstood (EF) | unknown |
| 50 | forgetting | forgetting (EF) | unknown |
| 54 | forgetting | forgetting (EF) | unknown |

> **\*** Q94 and Q45, Baseline: reasoned to the correct answer but emitted a wrong answer first; the automated scoring thus counts these as wrong answers. Similarly to an extraction failure, however, their reasoning can still be evaluated independently of this mishap. These dual-answer self-contradiction cases occurred other times in BL responses; however, these were the only instances wherein the non-emitted answer was correct.

### Outcome Distribution Overall

| Outcome | OS | CoT | BL |
|---|---:|---:|---:|
| ✓ correct | — | 16 | 6 |
| reasoning | 15 | 11 | 1 |
| misunderstood | 13 | 7 | 1 |
| math | 6 | 1 | 1 |
| hallucination | 3 | 0 | 0 |
| forgetting | 2 | 4 | 0 |
| unknown | 0 | 0 | 30 |
| *extraction failures* | — | 18 | 2 |

> **Note:** The *extraction failures* row is orthogonal to the others; it counts cells flagged `(EF)` in the [Per-Question Breakdown Table](#per-question-breakdown) above.

### Outcome Distribution by Error Type

The following tables are results for CoT and BL, split by the OS error type to which they belong. Counts continue to ignore extraction failures. `n` is the number of OS failures of that type. Outcome columns mirror the row order, so same-type persistence sits on the diagonal (offset one column by `✓`). This mirroring excludes Unknown, since unknown OS results were not included in the cross-strategy comparison.

#### Chain of Thought

| One Shot Error | n | ✓ | Reasoning | Misund | Math | Halluc | Forget | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logical Reasoning | 15 | 5 | 8 | 1 | 0 | 0 | 1 | 0 |
| Misunderstood | 13 | 4 | 2 | 5 | 1 | 0 | 1 | 0 |
| Math | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| Hallucination | 3 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| Forgetting | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |

#### Baseline

| One Shot Error Type | n | ✓ | Reasoning | Misund | Math | Halluc | Forget | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logical Reasoning | 15 | 3 | 0 | 0 | 0 | 0 | 0 | 12 |
| Misunderstood | 13 | 2 | 0 | 1 | 1 | 0 | 0 | 9 |
| Math | 6 | 1 | 1 | 0 | 0 | 0 | 0 | 4 |
| Hallucination | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| Forgetting | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |


### Comparison Analysis

As previously mentioned in the beginning of the [Cross-Strategy Comparison](#cross-strategy-comparison) section, BL is largely useless to analyze, since the vast majority of its responses contain only the answer with no reasoning, thus making it impossible to evaluate the type of error that has occurred when it is wrong. Consequently, I will only discuss the OS / CoT comparison from now on.

I continue to ignore extraction failures in BL / CoT as I have been doing in this section.

There are two important takeaways from this comparison.

#### 1. Error type persists across strategies

In other words, the error type is more about the problem than the prompting strategy: certain questions trip the model up in similar ways regardless of prompt. This suggests model-level rather than prompt-level limitations.

If one examines the diagonal on the [outcome distribution by error type for CoT](#chain-of-thought), it is clear that for each category, overall, CoT tends to err in the same way as OS:
- 8/10 CoT errors on questions where OS had a logical reasoning error are also logical reasoning errors for CoT
- 5/9 for misunderstanding the question
- 2/2 for forgetting

For OS math and hallucination errors, there is only one CoT error each, and neither is the same type as the OS error, but the sample size of one each is insignificant, so that does not discredit this trend.

That error type persists across strategies (or at least between OS and CoT) is important because it means that we can assume that if CoT does particularly well on any of the OS error type categories, it is probable that it ran into the same issues as OS but was able to overcome them, and is therefore better at that type of problem. Of course, the same could be said in reverse if CoT were to perform worse on a given type of problem.

If error type did not persist across strategies, CoT's performance on each of the OS categories would not be helpful at all, because it would just be distributed the same as in the overall distribution. The overall distribution could still be of some use—we might still be able to detect some patterns there—but we would lose out on one dimension of analysis. So this is great to see.

#### 2. CoT is great at math and (possibly) at not hallucinating

As mentioned above, CoT resolved 5/6 of OS's math errors. That is an 83% correction rate compared to 33% for reasoning and 31% for misunderstanding the question. CoT also made just one math error across all 39 problems, though with most of those not being math problems to begin with, given error type persistence, that says little on its own. Overall, CoT markedly improves math performance on this set. This makes sense, since showing your work and thinking step-by-step is a classic way to improve a person's performance at math, too. It is easier to verify intermediate calculations when they are performed explicitly. Skipping intermediate calculations is more likely to lead to an error. Given the strength of this logic and strong performance on the (albeit small) sample, it is likely that this improvement generalizes beyond these 6 problems.

Hallucinations in this sample also improve substantially, with 2/3 of the OS hallucination problems resolved in CoT. There are zero hallucination errors among the 39 problems for CoT (although this has the same caveat as with math of reduced weight given error type persistence). One possible mechanism for hallucination improvement by CoT is that it is during a skipped intermediate step that the model hallucinates, and forcing step-by-step thinking grounds the model in those steps, preventing the error from occurring. But this is more of a stretch than the logic for math problems, and given tiny sample size of 3, this data is suggestive but not conclusive that CoT improves performance on hallucination-inclined problems compared to OS.

## Refinement Attempts

Given the results of the [Comparison Analysis](#comparison-analysis) above, I thought that integrating CoT into OS could produce a drastic improvement in overall accuracy by strengthening math and (perhaps) hallucination performance while still retaining OS's excellent ESR.

While even those two categories added together are fewer in number than either logical reasoning errors or misunderstanding the question alone (9 vs. 15 and 13), the path forward here based on the data is much more informed than the path forward for improving either of the latter two categories. In other words, it is the low hanging fruit.

Furthermore, regardless of impact on these two particular categories, I already knew that CoT has a better overall AoES than OS does from initial summary data (see: [Summary Comparison](#summary-comparison)), so that is another reason to think that combining the two would be fruitful.

### One Shot and Chain of Thought

My first idea was to simply put both the OS and CoT prompts in with no other changes. The text of the prompts do not conflict and can easily coexist—no reason to complicate it. This became OS-and-CoT. See the [Sample Results](../README.md#sample-results) section of the README for an example of the OS-and-CoT prompt.

My only fear was that based on past experience, one has to be careful with overloading smaller models with too many instructions. The more you add, the more likely each piece is to be lost. As mentioned in the [Initial Analysis](#initial-analysis), this is what seems to happen to CoT—the CoT instruction makes it harder for the model to remember the formatting instruction, and consequently, the ESR plummets.

However, my hope was that given that OS manages to retain a very good ESR despite having a much larger prompt than CoT, that that would be the predominant force even when adding the CoT instruction. (As mentioned in the [Initial Analysis](#initial-analysis), I theorize that OS has a great ESR because it has a two-shot formatting instruction as opposed to BL and CoT's one-shot formatting instruction—all 3 have the "For example: '#### 42'" in the base prompt, but OS also demonstrates appropriate answer formatting a second time naturally as part of the example question and answer.)

#### OS-and-CoT Results

The results did not go as I expected. OS-and-CoT was neither the best of both worlds nor the worst, nor somewhere in the middle, nor some other clear combination. This is a more interesting finding than if it had worked as expected.

| Strategy | Accuracy | ESR  | AoES |
|---|---:|---:|---:|
| **OS-and-CoT** | 55.9% | 97.0% | 57.6% |
| **OS** | 51.9% | 90.7% | 57.2% |
| **CoT** | 37.0% | 51.9% | 71.2% |

> **Note:** The summary results above can also be found in the [Initial Analysis](#initial-analysis) and [here](../README.md#summary-comparison-between-prompting-strategies) in the README, where the other prompting strategies are also included.

The ESR was even better than I could have hoped for, at an extraordinary 97%, substantially better than ordinary OS. Indeed, this carries OS-and-CoT to being the strongest strategy overall, which is great. However, the AoES was almost identical to that of OS; the benefits of CoT were not carried over.

This raises two questions:
1. Why does the CoT prompt not exert obvious influence anywhere? All the stats are clearly much closer to those of OS than of CoT.
2. Why is the ESR even better than OS's, instead of being the same or dragged down by CoT?

#### 1. Why no apparent CoT influence?

I assert that this is simply because concrete examples dominate abstract instruction. The two signals compete, and OS wins out. Even if the CoT instruction is considered, the model might reasonably assume that the OS example is one that fulfills the instruction to think step by step.

Empirical results demonstrate this. Consider the responses to GSM8K question 0 for each prompting strategy below:

| Question |
|---|
| Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? | 

| Strategy | Model Answer |
|---|---| 
| **OS** | `Janet's ducks lay a total of 16*3 = <<16*3=48>>48 eggs for breakfast.\nShe also uses up 4*3 = <<4*3=12>>12 eggs to bake muffins for her friends.\nSo Janet has 48-12-3 = <<48-12-3=33>>33 eggs left over.\n\nTherefore, Janet makes 33*2 = <<33*2=66>>66 dollars every day at the farmers' market.\n\n#### 66` | 
| **CoT** | `To determine how much Janet makes from selling the eggs at the farmers' market every day, we need to follow these steps:\n\n1. **Calculate total eggs laid per day**: \n   - Janet's ducks lay 16 eggs per day.\n\n2. **Determine the number of eggs eaten for breakfast**:\n   - Janet eats 3 eggs for breakfast every morning.\n   \n3. **Calculate remaining eggs after breakfast**:\n   - Remaining eggs = Total eggs - Eggs eaten for breakfast\n   - Remaining eggs = 16 - 3 = 13\n\n4. **Determine the number of eggs used to bake muffins**:\n   - Janet uses 4 eggs to bake muffins for her friends every day.\n\n5. **Calculate remaining eggs after baking muffins**:\n   - Remaining eggs = Remaining eggs after breakfast - Eggs used for muffins\n   - Remaining eggs = 13 - 4 = 9\n\n6. **Calculate earnings from selling the remaining eggs**:\n   - Janet sells each egg for $2.\n   - Earnings = Remaining eggs * Price per egg\n   - Earnings = 9 * $2 = $18\n\nTherefore, Janet makes $18 every day at the farmers' market.\n\nFinal Answer: $#### 18` | 
| **OS-and-CoT** | `First find how many eggs Janet eats for breakfast: 3 eggs/day * 5 days = <<3*5=15>>15 eggs.\nThen subtract that number from the total number of eggs laid to find out how many are left over: 16 eggs - 15 eggs = <<16-15=1>>1 egg.\nNow multiply the number of leftover eggs by the price per egg to find out how much money she makes: 1 egg * $2/egg = $<<1*2=2>>2.\n\n#### 2` |

> **Note:** The entire result (i.e. the first line from the result output `jsonl` file) for each prompting strategy can be found in the README [here](../README.md#sample-results).

Observe how similar the responses of OS and OS-and-CoT are when compared to CoT's. Three notable points:
1. Most obviously, OS and OS-and-CoT are much shorter than CoT, which is certainly indicative of less step-by-step thinking.
2. More granularly, both OS and OS-and-CoT have GSM8K style math notation. OS has `<<16*3=48>>48` and OS-and-CoT has `<<3*5=15>>15`. This is not found in the CoT response.
3. CoT's response is more structured, each step having both a title and a number.

Clearly, in this example, the OS instruction has a much larger effect on how the model responds than the CoT instruction does, to the point that the former overrides the latter. So it is unsurprising that at an aggregate level OS-and-CoT performs similarly to OS, lacking any similarity to CoT results.

In my opinion, this is a case where the model behaves similarly to a human. If told to think step by step and given an example of how to respond, one assumes that that example is a good example of how to respond step by step (and certainly, the GSM8K-style response *is* step-by-step, just not to the extent of the CoT responses). Furthermore, and perhaps more importantly, following an existing format is just much easier than pioneering one's own. If I were taught a new mathematics concept, the theory is all well and good, but it does not truly make sense without an example, and the example is what I would keep looking back at when trying to solve my own problems.

This result also goes to show that stock GSM8K answers are not "full" chain of thought. Certainly, there is some level of that, but it is not at the level of detail that the model presumes should be done for CoT without an example (and interestingly, most of the BL responses that do have reasoning as opposed to being just the answer are closer to the CoT style, as well).

In short, the OS instruction not only fixes the output format, but also provides a structure for how to respond that is not like how the model responds when asked to do CoT without an example. And if you do not have to think of a format, why bother? Use the format you are given.

#### 2. Why is the ESR so good?

The only possibility that occurs to me is that the step-by-step thinking instruction makes adding the "#### " before the answer feel like a natural final *step*, leading the model to view it as more important.

Initially, I found this explanation to be unsatisfactory. It assumes that the model remembers to handle things step by step at all, and yet considering that we do not see the more expected results from CoT characteristic of thinking step by step, it seems presumptuous to conclude it would work in this new way.

However, the conclusions of question one actually reinforce this theory. They explain why we lose all the other aspects of CoT—the abstract CoT instruction is drowned out by the concrete OS example. If anything, the CoT instruction perhaps reinforces the OS behavior, insofar as it is step-by-step. But part of that reinforced behavior is the ESR, so while most of the expected effects of CoT drown out, this improved ESR surfaces.

### One Shot of Chain of Thought

Given how forcefully the OS example anchors the model's behavior, as demonstrated by the results in the above section, it was clear to me what to try next. Instead of using the first GSM8K training answer for the OS example, I would use a different answer strongly CoT in nature.

A compelling means to accomplish this suddenly struck me. I temporarily modified my code to pass the GSM8K training questions to the model rather than the test questions, and then ran an iteration with the CoT prompt and a question count of 1. I then took the model's output from that as the OS example answer. From here I derive the name: the OS example is of a CoT response. As a result of this approach, the example style unequivocally matches the abstract instruction.

#### OS-of-CoT Results

| Strategy | Accuracy | ESR  | AoES |
|:---|---:|---:|---:|
| **OS-and-CoT** | 55.9% | 97.0% | 57.6% |
| **OS** | 51.9% | 90.7% | 57.2% |
| **CoT** | 37.0% | 51.9% | 71.2% |
| **OS-of-CoT** | 26.5% | 38.7% | 68.4% |

This did not go well. OS-of-CoT is the worst of the four. However, poor performing results are not inherently a bad thing. From a research / scientific perspective, this is interesting and this is progress.

Insofar as making it more like CoT than OS-and-CoT is, this was definitely a success. The AoES is clearly very similar to that of CoT as opposed to those of other prompting strategies. Unfortunately, the ESR is also closer to CoT's than that of any other prompting strategy, but only in that CoT is now the second worst and OS-of-CoT is the worst—the difference is actually rather large. This weak ESR reduces the overall accuracy equally precipitously.

Like with the [results of OS-and-CoT](#os-and-cot-results), the OS-of-CoT results present us with two questions.
1. Why is the ESR so much worse than even CoT's?
2. Although it is a small difference (2.8%), why is the AoES worse than CoT's?

#### 1. Why is the ESR so bad?

Here we are presented with the opposite question of OS-and-CoT. I was quite surprised that there was such a bad ESR. Given that OS and OS-and-CoT both have great ones, and that the main thing they have in common is OS, I assumed that OS where just the answer is different would also do well on extraction.

But I hypothesize that this is simply due to the longer answer example. With a larger answer, the "#### " before the final answer is proportionally smaller and is therefore a weaker format signal.

#### 2. Why does OS-of-CoT have a slightly worse AoES than CoT does?

First, minor changes in prompt *phrasing*—let alone in prompt strategy—can produce differences in performance. See more on this in the [Formally Testing Previous Informal Tests](#formally-testing-previous-informal-tests) subsection of the [Limitations and Future Direction](#limitations-and-future-direction) section. So the difference here is within expected noise. Further, this is not a like-for-like comparison anyway, because the set of successfully extracted problems is different for different prompting strategies. Perhaps, for example, OS-of-CoT is slightly better at producing an extractable answer for problems that are harder (which would be a strength of the strategy masquerading as a weakness). This could be eliminated by recomputing accuracy on the subset of questions where extraction for both strategies was successful. It is also possible that the OS example induces rigidity in the model's behavior, locking it into imitating the example's approach, and some small subset of problems benefit from CoT's greater flexibility, but this would be hard to test. Overall, what is truly noteworthy here is that their performance is so *close*, not that there is a difference.

## Tradeoffs

Take another look at the summary comparison table.

| Strategy | Accuracy | ESR  | AoES |
|:---|---:|---:|---:|
| **OS-and-CoT** | 55.9% | 97.0% | 57.6% |
| **OS** | 51.9% | 90.7% | 57.2% |
| **CoT** | 37.0% | 51.9% | 71.2% |
| **OS-of-CoT** | 26.5% | 38.7% | 68.4% |
| **AO** | 8.9% | 97.4% | 9.2% |
| **BL** | 7.6% | 92.7% | 8.2% |

The key takeaway is that there is a tradeoff or tension between ESR and AoES. Of course, the tradeoff is not equivalent in all cases; otherwise, the overall accuracy would be consistent between strategies. But every strategy has a strong and weak trait between these two, and it is hard to increase one without decreasing the other, or at best keeping it the same.

As for why this tradeoff exists, all of the analysis in previous sections points to the lever being to what extent the model thinks step by step in its response. More step-by-step thinking of course means better AoES—the fundamental principle of CoT—but it comes at a cost. Perhaps this is due to a decreased capacity for the model to hold on to the instruction for how to format for extraction, or perhaps longer outputs just have more opportunities to drift from format.

This effect is most obvious in the cases of AO and BL. AO is explicitly instructed to exclude any thinking, and BL, lacking any instruction to do so, deigns not to. With this minimum possible degree of step-by-step thinking, both exhibit horrific AoES but excellent ESR.

Conversely, CoT and OS-of-CoT are the strongest on AoES and suffer on ESR. And, indeed, their responses have the most verbose step-by-step thinking. Of course, CoT has a much stronger ESR than OS-of-CoT—see [this subsection](#1-why-is-the-esr-so-bad) of the OS-of-CoT results analysis for more on that.

OS and OS-and-CoT strike the strongest balances, and thus top the leaderboard. This suggests diminishing returns of ESR and AoES as degree of step-by-step thinking decreases or increases, respectively. OS and OS-and-CoT boast ESRs very similar to those of AO and BL, despite middling degrees of step-by-step thinking, which is nonexistent for AO and BL. Yet OS and OS-and-CoT also retain most of the gains to AoES possible from step-by-step thinking. Moderate step-by-step thinking buys most of the AoES benefit while costing almost no ESR. 

## Limitations and Future Direction

I have alluded to or explicitly mentioned many limitations and future directions throughout the rest of this analysis. I will reiterate and expand on those here, but will also discuss topics not explicitly related to any of the above sections, as well.

I combine limitations and future direction into one section because there are many limitations that naturally inform a future direction that could resolve or mitigate them. Of course, there are both limitations and future directions not in that cross-category, as well, but for simplicity's sake I combine them all.

The expanding landscape of analysis is practically infinite. I could list potential future direction in this project for twice as long as the entire rest of this document and still have plenty more to propose. So I will do my best to keep it limited to some of the most intriguing possibilities across a diverse range of contexts and methodologies.

### Depth of Existing Analysis

To start, I only scratched the surface on the volume of analysis that could be done, even on the data I already have, even adhering to analysis techniques I have already used, and even only in contexts in which I have already performed them. This is a time limitation. While manual analysis does get faster with experience, it is still a significant time investment, and only a certain volume is tenable.

The clearest such example is as follows. Instead of performing the [Error Taxonomy](#error-taxonomy-and-distribution) informative to much of this analysis on 50 problems (and only 39 for CoT and BL), I could have done it on all 1319. Needless to say, this increased sample size would yield much greater confidence in related results. For example, as noted in [Distribution](#distribution), 'When considered, "How do forgetting errors tend to look?", it cannot be reliably determined from this data, since we only have a sample size of 2 for them.' Even if I had still only examined errors for OS, of which there are 512, that is a 10x increase to sample size. Examining 20 forgetting errors would be much more informative. Analysis of a larger set of problems might produce other unknown results, as well, such as revealing additional error types. Given that some error types are as infrequent as 2 or 3 out of 50, it is certainly plausible that other types could be obfuscated by the restricted sample size.

Another such case is the n=1 sample size for response-style comparison in the [Why no apparent CoT influence?](#1-why-no-apparent-cot-influence) subsection of the OS-and-CoT results analysis. Even if I had not expanded the overall analysis as suggested above, if I had examined all 39 problems for these questions, stronger conclusions would be able to be drawn regarding the patterns of response style by prompt strategy.

### The Uncertainty of Taxonomy

Taxonomy has fuzzy boundaries. It is often unclear what the true source of an error is and it is also possible to have multiple errors in a single problem.

#### The ambiguity of the true source of an error

As mentioned in the [Unknown](#unknown) section, sometimes it is not clear which category an error fits into. That section does not mention, however, that much of the time, even errors I did categorize could actually be closer to another type. It is a matter of confidence level. All we have is the model's output; we cannot see its internal state and do not know if it went wrong somewhere that is invisible. Consequently, some level of assumption is required.

Notably, math errors are the one category where a positive identification is high confidence, because a visible math error in the response is clearly not anything else. The reverse, however, does not hold. The absence of a visible math error does not rule one out, since an invisible miscalculation could still manifest as, for example, a seemingly hallucinated number.

The line is often particularly blurry between logical reasoning and misunderstanding the question, such as in questions 5 and 15, both of which I declared as the former. One could argue that on question 5, the model's failure was that it thought the discount on the second glass applied to all glasses after the first instead of every other, rather than the logical failure of not implementing the alternation correctly. On question 15, where the model essentially answers a different question, perhaps that is what it understood the question to be, rather than making a misstep during the logic of answering the first.

The fuzziness is also especially pronounced between misunderstanding and forgetting. I decided that questions where the model fails to acknowledge an important detail in the prompt (as in question 94) is a case of misunderstanding. Although I decided that forgetting errors only occur when something was mentioned in the response and then later not included where it needed to be, it would be reasonable to label the situations like question 94 as forgetting as well. It would also be reasonable to even create a third designation for this midpoint category, keeping forgetting as described and misunderstanding as its other subcategory: replacements rather than dropping, like six years older becoming six years younger (question 21).

Ultimately, there is fundamental subjective uncertainty in taxonomy, which is a weakness, albeit an unavoidable one. One can only use that which is observable. Much as a scientist studying a mouse's behavior has only that to go on, and not its internal experience, so too the model's output is all I have. The scientist could dissect the mouse's brain to go deeper, much as the model's behavior could be better understood with interpretability study of its thoughts outside the response, but even then, there remain practically endless layers of whys. Why did the model get this wrong? Well it misunderstood the question. Why did it misunderstand? Etc. Perhaps I should have been more conservative and labeled more responses as unknown, but I had to choose to cut it off at some point and make a call, and my cutting off point is what we can see and surmise based just on the output.

#### The presence of multiple errors

Additionally, a model's response sometimes contains multiple errors. For example, in question 31, in which characters are guessing the number of jelly beans in a container, one character says 80, and "another says 20 more than half the first one." The model instead calculates 80 more than half the first one ("80 + (1/2)*80"), having presumably misread the question. The third character predicts "25% more than the first one," which the model says is 90, having incorrectly calculated 25% of 80 as 10.

In these cases, I usually labeled the failure by the earliest error that makes the answer unrecoverable, or, if it was unclear what type of error that was, then the clearer one. There is no particular reason to choose the first one, and the cleanest analysis would instead acknowledge both error types. But that gets complicated quickly. My labeling system seems to me to be the most reasonable choice given that the error taxonomy process is extremely time consuming, so the alternative would be to lower the sample size even further. But I would be remiss not to mention the lack of a more robust classification system considering multiple errors per problem as a limitation and its potential for future direction.

### Tradeoff Resolution

As discussed in [Tradeoffs](#tradeoffs), there is a fundamental tension between ESR and AoES. This cannot be trivially resolved with the current model. So it would be interesting to try a larger model to see how it performs. I would first try Qwen2.5-**3B**-Instruct to keep consistent all variables other than size when compared to my current Qwen2.5-**1.5B**-Instruct. Performance would very likely increase across the board, and much further analysis could be done looking at how it comparatively improved across error types and such, but in this context, it would be interesting to see if it resolved this tradeoff. If not, perhaps there is some model size threshold at which it is overcome, at least in the context of GSM8K, and other sizes of Qwen2.5 instruction tuned models could be tested sequentially to find that.

As I said, the tradeoff cannot be *trivially* resolved with the current model, but perhaps there are other non-trivial ways to resolve this aside from a new model. Three ideas that occur to me are:
1. Few-shot and few-shot variations. It seems very likely that this would improve ESR beyond that of OS, OS-and-CoT, and OS-of-CoT. Perhaps it would also improve AoES by providing a further range of example answers and thus constrain the model to a lesser degree. I would predict that the degree to which this improvement would occur would scale with the number of few-shot examples (i.e. three-shot's performance > two-shot's, etc.).
2. Swap the OS example for a harder, multi-step problem to test whether example difficulty anchors reasoning depth. This could lead to a worse ESR, if it is true that a longer answer example buries the extraction example, as hypothesized in the [Why is the ESR so bad?](#1-why-is-the-esr-so-bad) section of the OS-of-CoT analysis. But it also seems plausible that a more difficult example could improve AoES without weakening ESR.
3. More advanced post-processing of responses to improve ESR. One option could be to try using a large frontier model to parse the responses of the smaller model for the answer. However, I find this less compelling and worthwhile since it is more so advanced string parsing than advanced interaction with the model.

### Prompt Adjustments

#### Wholly New Ideas

In the [Logical Reasoning Error](#logical-reasoning-error) section, I explained the error of question 41, which suggested a safe range from a dragon being -200 feet. Similarly, in the [Misunderstood Question](#misunderstood-question) section, I mentioned the problem where the model assumes that 16 oz cans of tomatoes refer to the volume after reduction rather than before. These sorts of problems reveal a certain lack of critical thinking by the model. It does not consider that one of its conclusions simply makes no sense. Now, in fairness, it might think that the questions are not expected to have answers that must make sense. But it does not mention considering that in its responses. Perhaps results could be improved by asking the model to consider whether its ultimate conclusion seems logical in isolation, and, if not, to redo the problem with that in mind. The model might then reflect that -200 feet does not make any sense as a concept. I sometimes notice this kind of self-checking in Claude's thinking, so at least one more capable model exhibits this behavior. Whether that emerges through training, system prompts, or simply larger scale is impossible to conclude from this observation alone, but it does demonstrate that it is an achievable ability.

As mentioned in the [Distribution](#distribution) section, I could try reducing the frequency of misreading the question in various ways.
> For example, the model could be told to restate the problem or even just to ensure that it understands the question, or similar such additions. These would likely make the model pay more attention to the question which I suspect would improve performance.

#### Formally Testing Previous Informal Tests

Early in the project, I tested adjustments in my prompts to find the most effective variants in an informal and untracked manner. This could be undertaken a second time using a formal approach.

First, there is the presence of the formatting one shot used in all prompts other than AO ("For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'"). I found that this improved ESR when initially building my prompts,  which is not surprising, since it is a one-shot example for formatting. It would be informative to test this again and gather the statistics on how it performs versus not using it and between variations of it.

The exact phrasing of my prompts also went untested. Before going further, I should note that although there are undoubtedly industry standard prompt phrasings that are more effective than mine, for the purposes of this project, I thought it most appropriate to perform the iteration myself. In any case, I was astounded by the degree to which very slightly different phrasing with the same meaning could produce drastically different results. This was the case both with respect to the formatting instruction mentioned above and the other elements of the prompts. In the beginning, I tried a few variations and picked the ones that seemed the best based on the summary results, which I did not keep, because I had already decided it was out of my intended scope for the project. This iteration produced decent prompts, but it left no historical trail of development to qualitatively analyze, nor did it result in any data that could quantitatively inform why it performed better (i.e. through comparing error type distributions), unless that reason was an improved ESR.

### Conclusion

As I said at the start of this section, there is a neverending list of possible ways to analyze this data or conduct new experiments to gather further data. This gives me a deeper appreciation for the analysis work of scientists. You cannot do it all. It is one thing to understand that conceptually when reading a research paper, but it is another to experience it myself and see the vast untrodden wilderness of potential research expanding before me, knowing how small my clearing of completed work is comparatively. But this also gives me hope that as AI continues to develop and can probe the problem space at a rate far faster than we can, it will propel us forward across all realms of data analysis and science more broadly at an unprecedented and presently unfathomable pace. Although I appreciated the enormous impact of AI in theory, and from afar, like through the achievements of AlphaFold, to have it so plainly obvious in the context of my own project hammers it home so much more deeply.
