Summary for now:
Math: 9
misunderstood question: 11
logical reasoning: 13
hallucination: 4
extraction: 10

My revised counts:
Math: 6
logical reasoning: 15
misunderstanding: 13
hallucination: 3
forgetting: 2
extraction: 10
unknown: 1

just going to discard the secondary point for now and acknowledge that it exists in my analysis with an example or two

I want to take the numbers with a grain of salt, since as I said earlier, the overall rate for extraction failures in the full dataset was 10%, and it's 20% here, which is very different.

I should write about how I already added the guide on how to extract with an example. Compare those.



Of course to some extent without further diving into interpretability, you can't fully know. Perhaps a model misreads a question but it appears as a logical reasoning error in the problem. Or they do math wrong but don't write it down and then it appears like... logical reasoning? (hmm maybe thats why there is a lot of reasoning errors? but i digress). so you never really know. but we're just going off what we can observe. much as i scientist watching a mouse's behavior has all that to go on, and not its internal experience, so too this is all i have (though of course in theory a much deeper dive is possible). But continuing the metaphor to animal brains, the thing is, there is a why for every mistake? Why did they get this wrong? Well they misread. Well why did they misread? Etc. So you have to choose the cut it off at some point, and my cutting off point is what we can see in the output and surmise from.

there are not obviously clean boundaries between categories

obviously we can see not a perfection sample, unsurpirisingly, since double the rate of extraction failures of the full set (which had a 90% extraction rate), but it's still something

Use the categories by where the first decisive failure happens. Don’t try to label every flaw; label the earliest/main flaw that makes the answer unrecoverable.

of course the thing is that especially for extraction failures, that doesn't mean solving that will make it right. there are plenty of extraction failures where what was failed to be extracted was a wrong answer. For example, question 6. Less obviously, this is also true for other problems, though with a lesser frequency. Sometimes it starts bad and gets worse. For example, 33. Or 35.

Also often it's not totally clear which category it fits into. The only source of info is the model's response, and sometimes that is insufficient to categorize. For example, problem 42. the model could have the info and then use it wrong or could have filed the info wrong in the first place. I feel like it's hard to say based on what the model says. In these cases, I just categorize it as "unknown."

One thing that strikes me is that the model doesn't consider whether the numbers make sense. Like question 49, where it thinks the 16 oz can is a measurement after reduction rather than a measurement before reduction, which of course makes no sense, why would a can be labeled by some amount it is ultimately reduced to be the cook. Or even more egregiously, Question 21 is most striking in this regard, suggesting a distance of -200 feet, which obviously doesn't make any sense. 

1. &nbsp;
   - logical reasoning error
   - wrong numbers interacting in so many ways. come back to this one

2. &nbsp;
   - revised: reasoning error
   - Math error
   - interpreted increase of 150% as 1.5x not 2.5x

3. &nbsp;
   - extraction error
   - no ####

4. &nbsp;
    - logical reasoning error
    - first two glasses counted as just one. all the remaining as 60% off rather than every other

5. &nbsp;
    - revised: reasoning error
    - Math error
    - first 40% interpreted as (1 - .4) aka 60%

6. &nbsp;
    - extraction error AND unknown
    - no ####
    - i truly cannot comprehend what's happening here

7. &nbsp;
    - misunderstood question
    - Thought the cost was per item not per dozen (extra *12)

8. &nbsp;
    - Arithmetic mistake
    - (7 * $1.5) - $3 = $6

9. &nbsp;
    - extraction error
    - no ####

10. &nbsp;
    - reasoning error
    - total value jewelery - total value electronics instead of max profit

11. &nbsp;
    - revised: reasoning
    - hallucinated steps?
    - answers as though the remaining 6 should be done at 4mph not the total 12

12. &nbsp;
    - revised again: reasoning
    - Revised: math or reasoning?
    - Math error?
    - I'm very unsure about this one. It seems like it does an extra -1 on the second line preemptively for no reason. It doesn't do the math it has written down incorrectly, but there is invisible math going haywire.

13. &nbsp;
    - misunderstood question
    - The model thought Raymond was 6 years younger not 6 years older than Samantha

14. &nbsp;
    - misunderstood question
    - The model thought the boots were $5 less than the shoes together not $5 more

15. &nbsp;
    - extraction error
    - no ####. A lot of math in this one

16. &nbsp;
    - math
    - also misreading, calculating 2nd person as 80 + half of 80, not 20 + half of 80
    - 80+(1/2)\*80=120>>120, 80+25%\*80=90>>90

17. &nbsp;
    - extraction error
    - no ####. Like 15. A lot of math in this one

18. &nbsp;
    - revised: hallucination. 60 came from nowhere
    - not sure, come back to this one

19. &nbsp;
    - reasoning error I think
    - adds $5 to total of lego set money rather than total of video game cost

20. &nbsp;
    - extraction error
    - no ####

21. &nbsp;
    - reasoning error I think?
    - 1000-1200 instead of 1200-1000

22. &nbsp;
    - misunderstood question
    - thinks there are 250 calories per bag rather than per serving
    - plus other stuff

23. &nbsp;
    - misunderstood question
    - The questions says 1 lb of beeswax and wicks are $10. Model says it's $10 per 2 lb and no info for wicks

24. &nbsp;
    - math
    - she wrote 5 + (2/5)*5 = <<5+(2/5)*5=6>>6 articles

25. &nbsp;
    - revised: reasoning
    - ???
    - i dont even know. misunderstood question or reasoning

26. &nbsp;
    - hallucinated steps
    - made up that he bought 3 blue ties

27. &nbsp;
    - revised: forgetting
    - logical reasoning?
    - forgot to account for week vs day. knew it was per day but then thought it was per week next step

28. &nbsp;
    - extraction error
    - no ####

29. &nbsp;
    - Revised: forgetting
    - logical reasoning? misunderstood question?
    - they forgot information. they had the nubmer of kittens born but not the ones taken from the shelter

30. &nbsp;
    - extraction error
    - no ####

31. &nbsp;
    - Revised: reasoning
    - hallucinated steps
    - thought they needed to add 15 lbs to the truck / driver for some reason

32. &nbsp;
    - revised: misunderstood question
    - similar to 29. logical reasoning? misunderstood question? thought delivery fee was 25% not general on top and then dropped delivery.
    - model missed the $3 delivery fee. not forgetting because it never brought it up

33. &nbsp;
    - misunderstood question and also math
    - model missed the sour and then also calculated 20% of 25 wrong

34. &nbsp;
    - revised: misunderstood question
        - thought it was a salary that would change by the %. Honestly understandable, took me a while to get it
    - i dont understand this question myself, come back

35. &nbsp;
    - revised: reasoning
    - logical reasoning?
    - doing weird things throughout but initial one is dividing by 2 twice (x6 (rather than 12), /2). Then later misunderstood question

36. &nbsp;
    - revised: reasoning
    - hallucinated steps?
    - interprets 4 gallons as both 4 gallons and 4 additional miles

37. &nbsp;
    - misunderstood question
    - seems to think it's number of hours reading rather than number of hours of both

38. &nbsp;
    - revised: misunderstood question
    - misunderstood question? logical reasoning?
    - missing 5 people of one of the two teams per school

39. &nbsp;
    - math
    - 175+140+280=695>>695

40. &nbsp;
    - math
    - 750*2%=150>>150

41. &nbsp;
    - logical reasoning
    - classic case of what is it even doing with the 110-180. dont know how else to categorize

42. &nbsp;
    revised: unknown
        - straight up unclear.  the model could have the info and then use it wrong or could have filed the info wrong in the first place. I feel like it's hard to say based on what the model says.
    - misunderstood question? logical reasoning?
    - ray has half as much as david instead of sarah by model's reckoning

43. &nbsp;
    - extraction error
    - no ####

44. &nbsp;
    - math
    - 4*125-2 = <<4*125-2=503>>503 pounds

45. &nbsp;
    - revised: hallucination
        - pulling the number 2 from nowhere
    - reasoning?
    - classic i dont even know what its doing here situation, could be something else though

46. &nbsp;
    - misunderstanding question
    - thinks it's 10% of monthly salary not yearly salary that it goes up by each year

47. &nbsp;
    - extraction error
    - no ####. Like 15. A lot of math in this one

48. &nbsp;
    - revised 2: misunderstanding. missed the 12
        secondary: forgot to add in dogs (forgetting) OR thought they already added in dogs (logic)
    - revised: Forgetting
    - logical reason? misunderstanding question? another one where they just lost some info, maybe should be its own category.
    - they miussed that 12 less than combined pet dogs and cats, just did the latter part of that to calculate rabbits

49. &nbsp;
    - misunderstood question?
    - thinks the 16 oz can is a measurement after reduction rather than a measurement before reduction

50. &nbsp;
    - logical reasoning
    - 25+20 instead of 25-20