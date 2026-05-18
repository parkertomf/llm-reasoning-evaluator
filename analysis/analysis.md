# Analysis

## Overall Results

Chain of thought had by far the best accuracy on extraction success, but also by far the worst extraction success rate.

One shot overall had the best accuracy due to its combined great extraction rate and accuracy on extraction success. I wonder if one shot combined with chain of thought would produce great results, but one has to be careful with overloading smaller models with too many instructions, in my experience.

It is striking that merely suggesting thinking step by step (chain of thought) causes such a drastic drop in extraction success rate compared to baseline (51.9% vs 92.7%). This goes to show the above point that it is easy to overload smaller models on instructions—add the ""Explain your answer step by step," and the "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'" is often forgotten.

Baseline is the worst, even worse than answer only. This is surprising since using its response to think through an answer typically makes a model perform better—that is the whole idea behind chain of thought, and answer only makes it answer without any of that thinking. Yet even ignoring extraction as a factor, answer only had better results than baseline: a 9.2% vs 8.2% success rate.

## Error Taxonomy and Distribution

## Cross Strategy Comparison

Export from Google Sheets

## Refinement Attempts

## Tradeoffs

## Limitations

Aside from the overall statistics, which are as good as it can get, my error taxonomy and evaluation of the 50 questions therein are limited by their small sample size.

## Future Direction

One idea that occurs to me is trying two-shot where in all permutations of stock and chain of thought examples (2 of one, one of each) and seeing what that does.

Of course, testing a different model would also be interesting to see if results are consistent, but that is a whole nother can of worms.