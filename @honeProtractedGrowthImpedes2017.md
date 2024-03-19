---
title: "Protracted growth impedes the detection of sexual dimorphism in non-avian dinosaurs"
authors: D. W. E. Hone, Jordan C. Mallon
year: 2017
---
# Protracted growth impedes the detection of sexual dimorphism in non-avian dinosaurs
#### 0.1.1 David W. E. Hone, Jordan C. Mallon
**Link**: https://onlinelibrary.wiley.com/doi/abs/10.1111/pala.12298
**DOI**: 10.1111/pala.12298
**Authors**: [[D. W. E. Hone]], [[Jordan C. Mallon]]
**Links**:
**Tags**: #paper

## 1 Summary
	Apart from the other difficulties with establishing sexual dimorphism, the growth pattern of dinosaurs throws a wrench in the works. Basically: dinosaurs show a pattern of slow, possibly indeterminate growth - they grow for basically their whole lives. This makes sexual size dimorphism hard to find, since a male will be the same size as a female a few years older (even if there is clear dimorphism among individuals of the same age).

## 2 Abstract
```
Evidence for sexual dimorphism is extremely limited in the non-avian dinosaurs despite their high diversity and disparity, and despite the fact that dimorphism is very common in vertebrate lineages of all kinds. Using body-size data from both Alligator mississippiensis and Rhea americana, which phylogenetically bracket the dinosaurs, we demonstrate that even when there is strong dimorphism in a species, random sampling of populations of individuals characterized by sustained periods of growth (as in the alligator and most dinosaurs) can result in the loss of this signal. Dimorphism may be common in fossil taxa but very hard to detect without ontogenetic age control and large sample sizes, both of which are hampered by the limitations of the fossil record. Signal detection may be further hindered by Type III survivorship, whereby increased mortality among the young favours the likelihood that they will be sampled (unless predation or taphonomic bias against small size acts against this). These, and other considerations relating to behaviour and ecology, provide powerful reasons to suggest that sexual dimorphism in dinosaurs may be very difficult to detect in almost all currently available samples. Similar issues are likely also to be applicable to many fossil reptiles, or animals more generally.
```

## 3 Notes
- Both clades that form the [[Extant Phylogenetic Bracket]] for [[Dinosauria]] ([[Aves]] and [[Pseudosuchia]]) have [[Sexual Dimorphism|sexually dimorphic species]]
- Lots of claims for [[Sexual Dimorphism]] in [[Dinosauria]], but all controversial - sample size, &c.
- One explanation is mutual sexual selection, but that still leaves some gaps
- Nearly all [[Aves|birds]] and [[Mammalia|mammals]] reach sexual maturity close to somatic maturity
	- Reach full ornamentation size as growth is ceasing
- Most other [[Vertebrata]] lineages spend most of their life below the asymptotic size
	- Thus: might be easy to identify dimorphism between members of the same age, but very difficult when you get a random mixture
		- Especially if you don't a priori know the age / sex!
- ![[Pasted image 20230523153812.png]]
- Their basic idea: there might be dimorphism in [[Dinosauria]], but we can't find it because of the issues with protracted growth and being unable to control for age.
- Tested this idea using the [[Extant Phylogenetic Bracket]]
	- American alligator and greater rhea
	- Both are strongly sexually dimorphic
- Alligators use a [[von Bertalanffy Equation]] to model length against age
- Rhea uses a [[Gompertz curve]]
- Replicated growth curves for males and females of each species
	- Simulated populations around this line using changes in the standard deviation as per the original papers (increased logarithmically with age)
- Then drew samples of various sizes to test whether they could recover the difference (even knowing the sex)
- Tried using a $t$-Test, and did the experiment 1000 times to calculat the probability of being able to recover the signal
- Next step: accounted for the different population structures (percent which are of various ages)
	- Used a sample population structure for the alligators - no mention of the rheas
- Also accounted for some taphonomic bias
	- In [[Dinosaur Provincial Park]], there is a strong bias against individuals <60kg - hence, they tried again with removing all of the individual <60kg (again, from the alligators)
 - ![[Pasted image 20230523192701.png]]
 - ![[Pasted image 20230523192714.png]]
 - Despite the fact that the dimorphism is stronger in the alligators, because the rheas spend more time at their maximum size, it is far easier to recover evidence of sexual dimorphism from them
 - In large part, this reflects not having to control for age
 - Rheas: ~15 per sex to recover dimorphism
 - Alligators: ~35 per sex
	 - at the $\alpha=0.05$ confidence level
	- When the <60kg ones are removed, this shrinks down to ~25 of each
- This number of specimens for [[Dinosauria]] are basically unattainable - even large bonebeds have around 50 individuals, not the >70 that would be needed to find dimorphism
	- Even worse: most bones, &c. are distorted so that the information is just not there
- This is basically the easiest way to detect dimorphism - we know the age and sex. There are tonnes of other issues that make practical application even more difficult
	- If there's trouble doing it under basically ideal circumstances, imagine it with all the difficulties!
- 

## 4 Next Steps / Ideas