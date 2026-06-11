# Analysis

Note that whenever reference is made to a question number, it is by its index in the test split of GSM8K, which corresponds to the `index` value for each result record `json` in a one of the `jsonl` output files, including in the first 50 errors file mentioned below in the [Error Taxonomy and Distribution](#error-taxonomy-and-distribution) introduction.

## Overall Results

### Summary Comparison

This summary table is replicated here from the [README](../README.md#summary-comparison-between-prompting-strategies) for the reader's convenience. The sole addition is prompt strategy acronyms.

| Strategy | Accuracy | Extraction Success Rate  | Accuracy on Extraction Success |
|:---|---:|---:|---:|
| **One Shot and Chain of Thought (OS-and-CoT)** | 55.9% | 97.0% | 57.6% |
| **One Shot (OS)** | 51.9% | 90.7% | 57.2% |
| **Chain of Thought (CoT)** | 37.0% | 51.9% | 71.2% |
| **One Shot of Chain of Thought (OS-of-CoT)** | 26.5% | 38.7% | 68.4% |
| **Answer Only (AO)** | 8.9% | 97.4% | 9.2% |
| **Baseline (BL)** | 7.6% | 92.7% | 8.2% |

#### Initial Analysis

OS-and-CoT has the overall best accuracy due to its excellent extraction rate and good accuracy on extraction success, with OS not far behind. See the [Refinement Attempts](#refinement-attempts) and [Tradeoffs](#tradeoffs) sections below for extensive commentary on OS-and-CoT as well as OS-of-CoT. Each of these were added as a result of error analysis.

CoT has by far the best accuracy on extraction success, but remains middling in overall accuracy due to a poor extraction success rate, at which it is only better than OS-of-CoT.

Merely suggesting thinking step by step (CoT) causes a drastic drop in extraction success rate compared to BL (51.9% vs 92.7%). This goes to show that it is easy to overload smaller models on instructions—add the "Explain your answer step by step," and the "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'" is often forgotten.

I suspect that the above does not occur for OS because while BL and CoT have a formatting-specific one-shot embedded in the prompt ("For example: '#### 42'"), OS's example response includes appropriate formatting as well, which makes it two-shot with respect to formatting, strengthening the model's extraction rate.

BL is the worst, even worse than AO, even when ignoring extraction as a factor. This is surprising since using its response to think through an answer typically makes a model perform better—that is the whole idea behind CoT as a methodology, and AO makes it answer without any of that thinking. However, as the [Cross-Strategy Comparison](#cross-strategy-comparison) section below reveals, BL rarely contained a response beyond the answer, either. But that only explains why BL does not do meaningful better than AO, not why it does meaningfully worse, so it remains surprising.

## Error Taxonomy and Distribution

Before having added OS-of-CoT and OS-and-CoT, of which I had not yet conceived, I wanted to analyze the errors of the existing prompting methods with the hope that doing so would inform how to refine the prompts to create newer, better ones. Of course, that's what led to OS-of-CoT and OS-and-CoT.

Since OS was the best performer among the original four in terms of overall accuracy, I chose to analyze its results to start.

I collected the first 50 errors from the OS results, using a script I wrote to assist (see [find_errors.py](scripts/find_errors.py) in the scripts directory), which parses a results file for a given number of regular errors / extraction failures and writes them to a new `jsonl` file. **A copy of the first 50 errors file can be found [here](2026-05-08_13-34-48_qc100_one-shot_first_50_errors.jsonl) in the analysis directory.** This may be of use for reference when specific problems are referred to throughout this document. Consider using a jsonl reader website like https://jsonl.co/ to make the file more readable.

I then manually categorized the 50 errors into the following categories. I initially had in mind the first four. The final three were additions I made during the process.

### Taxonomy 
1. Extraction Failure 
2. Math Error
3. Misunderstood Question
4. Logical Reasoning Error
4. Hallucination
5. Forgetting
6. Unknown

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

In some cases, it is not clear which category an error fits into. The only source of information is the model's response, and sometimes that is insufficient to categorize with confidence. For OS, there was only one case of this in the 50 problems I examined, question 77. In this case, the model does not give enough information for it to be clear whether it misread a fact or it misapplied logic to correctly-read facts; in other words, it is not clear whether it is an issue of miusunderstanding the question or logical reasoning. To make the point clearer, AO responses, assuming the model actually does give only the answer, are always categorized as unknown errors when they are wrong—there is not enough reasoning visible to make a reliable conclusion as to what type of error it is. While other prompting strategies do usually having reasoning, that does not mean there is enough information in their responses, either. See more on this in the [Limitations](#limitations) section below.

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

Misunderstanding the question is not far behind reasoning in count—close enough that it is certainly possible that in the full dataset, it is actually a more frequent type than logical reasoning errors. And I can readily imagine various ways to possibly improvement here. For example, the model could be told to restate the problem or even just to ensure that it understands the question, or similar such additions. These would likely make the model pay more attention to the question which I suspect would improve performance.

There is a drop-off before the frequency of math problems, but there are still enough of these that improving them would be useful. In contrast, hallucination and forgetting are both uncommon enough that trying to improve them is not worthwhile for now.

Improving extraction rate would certainly be helpful. After all, that is why OS has a better performance than CoT—its extraction rate is substantially better, enough to overcome its lower accuracy on extraction success. However, resolving an extraction error does not mean that the model will get the right answer. There are many extraction failures where what was failed to be extracted was itself a wrong answer, such as in question 8. Furthermore, I have already worked on improving the extraction rate informally as mentioned in the [Prompt Formatting](../README.md#prompt-format) section of the README.

Less obviously, the point above regarding fixing extraction not necessarily leading to a correct answer is also true for error types other than extraction failures—that is to say, sometimes, there are multiple other error types in one response (though with a lesser frequency than when one is an extraction failure). For example, question 60 includes both misunderstanding the question and then later a math error. Resolving just one half of that would still produce a wrong answer.

**A note on the sample size:** Compared to the overall 1319 set of test questions in GSM8K, 50 is a small sample size. However, one indication of good representativeness is that the extraction failure rate is very close to that of the rate in the full set. 10/50 or 20% of the failures in this sample are extraction failures compared to 123/635 in the full set or 19.4%. Of course, this does not necessarily mean that the proportions of the other error types are equally accurate, but it is nonetheless a good sign for the accuracy of the distribution in this sample relative to the full set. Crucially, though, this definitely does not mean that the sample size for each error type is necessarily large enough to make meaningful conclusions on it. When considered, for example, "How do forgetting errors tend to look?", it cannot be reliably determined from this data, since we only have a sample size of 2 for them. See more on this the [Limitations](#limitations) section.

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

> **\*** Q94 and Q45, Baseline: reasoned to the correct answer but emitted a wrong answer first; the automated scoring thus counts these as wrong answers. Similarly to an extracion failure, however, their reasoning can still be evaluated independently of this mishap. These dual-answer self-contradiction cases occurred other times in BL responses; however, these were the only instances wherein the non-emitted answer was correct.

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

Results for CoT and BL, split by the OS error type to which they belong. Counts continue to ignore extraction failures. `n` is the number of OS failures of that type. Outcome columns mirror the row order, so same-type persistence sits on the diagonal (offset one column by `✓`). This excludes Unknown, since unknown OS results were not included in the cross-strategy comparison.

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

In other words, the error type is more about the problem than the prompting strategy: certain questions trip the model up in similar ways regardless of prompt. This suggests model-level rather prompt-level limitations.

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

Given the results of the [Comparison Analysis](#comparison-analysis) above, I thought that integrating CoT into OS could produce a drastic improvement in overall accuracy by strengthening math and (perhaps) hallucination performance while still retaining OS's excellent extraction rate.

While even those two categories added together are fewer in number than either logical reasoning errors or misunderstanding the question alone (9 vs 15 and 13), the path forward here based on the data is much more informed than the path forward for improving either of the latter two categories. In other words, it is the low hanging fruit.

Furthermore, regardless of impact on these two particular categories, I already knew that CoT has a better overall accuracy on extraction success than OS does from initial summary data (see: [Summary Comparison](#summary-comparison)), so that is another reason to think that combining the two would be fruitful.

### One Shot and Chain of Thought

My first idea was to simply put both the OS and CoT prompts in with no other changes. The text of the prompts do not conflict and can easily coexist—no reason to complicate it. This became OS-and-CoT. See the [Sample Results](../README.md#sample-results) section of the README for an example of the OS-and-CoT prompt.

My only fear was that based on past experience, one has to be careful with overloading smaller models with too many instructions. The more you add, the more likely each piece is to be lost. As mentioned in the [Initial Analysis](#initial-analysis), this is what seems to happen to CoT—the CoT instruction makes it harder for the model to remember the formatting instruction, and consequently, the extraction rate plummets.

However, my hope was that given that OS manages to retain a very good extraction rate despite having a much larger prompt than CoT, that that would be the predominant force even when adding the CoT instruction. (As mentioned in the [Initial Analysis](#initial-analysis), I theorize that OS has a great extraction rate because it has a two-shot formatting instruction as opposed to BL and CoT's one-shot formatting instruction—all 3 have the "For example: '#### 42'" in the base prompt, but OS also demonstrates appropriate answer formatting a second time naturally as part of the example question and answer.)

#### OS-and-CoT Results

The results did not go as I expected. OS-and-CoT was neither the best of both worlds nor the worst, nor somewhere in the middle, nor some other clear combination. This is a more interesting finding than if it had worked as expected.

| Strategy | Accuracy | Extraction Success Rate  | Accuracy on Extraction Success |
|---|---:|---:|---:|
| **OS-and-CoT** | 55.9% | 97.0% | 57.6% |
| **OS** | 51.9% | 90.7% | 57.2% |
| **CoT** | 37.0% | 51.9% | 71.2% |

> **Note:** The summary results above can also be found in the [Initial Analysis](#initial-analysis) and [here](../README.md#summary-comparison-between-prompting-strategies) in the README, where the other prompting strategies are also included.

The extraction success rate was even better than I could have hoped for, at an extraordinary 97%, substantially better than ordinary OS. Indeed, this carries OS-and-CoT to being the strongest strategy overall, which is great. However, the accuracy on extraction success was almost identical to that of OS; the benefits of CoT were not carried over.

This raises two questions:
1. Why does the CoT prompt not exert obvious influence anywhere? All the stats are clearly much closer to those of OS than of CoT.
2. Why is the extraction rate even better than OS's, instead of being the same or dragged down by CoT?

#### 1. Why no apparent CoT influence?

I assert that this is simply because concrete examples dominate abstract instruction. The two signals compete, and OS wins out.

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
1. Most obviously, OS and OS-and-CoT are much shorter than CoT, which is certainly indicative of less step by step thinking.
2. More granularly, both OS and OS-and-CoT have GSM8K style math notation. OS has `<<16*3=48>>48` and OS-and-CoT has `<<3*5=15>>15`. This is not found in the CoT response.
3. CoT's response is more structured, each step having both a title and a number.

Clearly, in this example, the OS instruction has a much larger effect on how the model responds than the CoT instruction does, to the point that former overrides the latter. So it is unsurprising that at an aggregate level OS-and-CoT performs similarly to OS, lacking any similarlity to CoT results.

In my opinion, this is a case where the model behaves similarly to a human. If told to think step by step and given an example of how to respond, one assumes that that example is a good example of how to respond step by step (and certainly, the GSM8K-style response *is* step by step, just not to the extent of the CoT responses). Furthermore, and perhaps more importantly, following an existing format is just much easier than pioneering one's own. If I were taught a new mathematics concept, the theory is all well and good, but it does not truly make sense without an example, and the example is what I would keep looking back at when trying to solve my own problems.

This result also goes to show that stock GSM8K answers are not "full" chain of thought. Certainly, there is some level of that, but it is not at the level of detail that model presumes should be done for CoT without an example (and interestingly, most of the BL responses that do have reasoning as opposed to being just the answer are closer to the CoT style, as well).

In short, the OS instruction not only fixes the output format, but also provides a structure for how to respond that is not like how the model responds when asked to do CoT without an example. And if you do not have to think of a format, why bother? Use the format you are given.

#### 2. Why is the extraction rate so good?

The only possibility that occurs to me is that the step by step thinking instruction makes adding the "#### " before the answer feel like a natural final *step*, leading the model to view it as more important.

Initially, I found this explanation to be unsatisfactory. It assumes that the model remembers to handle things step by step at all, and yet considering that we do not see the more expected results from CoT characteristic of thinking step by step, it seems presumptious to conclude it would work in this new way.

However, the conclusions of question one actually reinforce this theory. They explain why we lose all the other aspects of CoT—the abstract CoT instruction is drowned out by the concrete OS example. If anything, the CoT instruction perhaps reinforces the OS behavior, insofar as it is step by step. But part of that reinforced behavior is the extraction rate, so while most of the expected effects of CoT drown out, this improved extraction rate surfaces.

### One Shot of Chain of Thought

Given how forcefully the OS example anchors the model's behavior, as demonstrated by the results in the above section, it was clear to me what to try next. Instead of using the first GSM8K training answer for the OS example, I would use a different answer strongly CoT in nature.

A compelling means to accomplish this suddenly struck me. I temporarily modified my code to pass the GSM8K training questions to the model rather than the test questions, and then ran an iteration with the CoT prompt and a question count of 1. I then took the model's output from that as the OS example answer. From here I derive the name: the OS example is of a CoT response. As a result of this approach, the example style unequivocally matches the abstract instruction.

#### OS-of-CoT Results

| Strategy | Accuracy | Extraction Success Rate  | Accuracy on Extraction Success |
|:---|---:|---:|---:|
| **OS-and-CoT** | 55.9% | 97.0% | 57.6% |
| **OS** | 51.9% | 90.7% | 57.2% |
| **CoT** | 37.0% | 51.9% | 71.2% |
| **OS-of-CoT** | 26.5% | 38.7% | 68.4% |

This did not go well. OS-of-CoT is the worst of the four. However, poor performing results are not inherently a bad thing. From a research / scientific perspective, this is interesting and this is progress.

Insofar as making it more like CoT than OS-and-CoT is, this was definitely a success. The accuracy on extraction success is clearly very similar to that of CoT as opposed to those of other prompting strategies. Unfortunately, the extraction success rate is also closer to CoT's than that of any other prompting strategy, but only in that CoT is now the second worst and OS-of-CoT is the worst—the difference is actually rather large. This weak extraction rate reduces the overall accuracy equally precipitously.

Like with the [results of OS-and-CoT](#os-and-cot-results), the OS-of-CoT results present us with two questions.
1. Why is the extraction success rate so poor?
2. Although it is a small difference (2.8%), why is the accuracy on extraction success worse than CoT's?

#### 1. Why is the extraction rate so bad?

Here we are presented with the opposite question of OS-and-CoT. I was quite surprised that there was such a bad extraction rate. Given that OS and OS-and-CoT both have great ones, and that the main thing they have in common is OS, I assumed that OS where just the answer is different would also do well on extraction.

But I hypothesize that this is simply due to the longer answer example. With a larger answer, the "#### " before the final answer is proportionally smaller and is therefore a weaker format signal.

#### 2. Why does OS-of-CoT have a slightly worse accuracy on extraction success than CoT does?

The most important point here is that this difference in results is small enough that it probably does not actually represent a difference in performance.

Initially, I had assumed that this must be a statistically significant difference given that 1319 questions is a substantial sample size, but I had failed to consider an important variable. Accuracy on extraction success is graded, obviously, only on the problems where the answer is successfully extracted. But what that means is that comparing accuracy on extraction success between prompting strategies is not a completely like-for-like comparison, because the set of successfully extracted problems is different for different prompting strategies.

So OS-of-CoT's accuracy on extraction success is graded on the 510 problems for which its answer was successfully extracted, and CoT's is graded on its 685 such problems. One might think that these sets are probably still representative, but we do not know that with certainty. It is possible, for example, that CoT is slightly better at producing an extractable answer for problems that are easier or that OS-of-CoT is slightly better at producing an extractable answer for problems that are harder. These differences would both explain the discrepancy here. And it is worth noting that the latter explanation would actually be the result of OS-of-CoT being better at something, yet appearing worse as a result.

In short, the gap here may be partly or wholly an artifact of comparing different question subsets. However, that is not necessarily the case, either. It could be a real small gap in performance.

I hypothesize the following as one possible explanation if it is a real difference. The OS example induces rigidity in the model's behavior, locking it into imitating the example's approach. This rigidity is not induced by the abstract CoT instruction. Some small subset of problems need an approach that the OS-of-CoT example does not demonstrate, and therefore benefit from the greater freedom allowed by CoT. So OS-of-CoT loses a few points on these problems. However, it would require much more data analysis to substantiate this hypothesis, but that is out of the current scope, so it remains untested and purely theoretical. See the [Future Direction](#future-direction) section below for more such matters.
