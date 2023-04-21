---
title: "Unlocking the Power of ChatGPT and AI in Test Automation: Next Steps"
authors: Anand Bagmar
date: 2023-04-13
series: Applitools Webinars
---

# Unlocking the Power of ChatGPT and AI in Test Automation: Next Steps

**Authors**: [[Anand Bagmar]]
**Series**: [[Applitools Webinars]]
**tags**: #talk #automation #test #ai

## Summary

## Abstract
```
Explore how to use a combination of AI-powered tools—including ChatGPT, GitHub CoPilot, and more—for your software test automation needs in this upcoming event. Through specific use cases, Anand Bagmar will equip attendees with the knowledge to enhance their test automation processes—making them faster, more stable, and easier to implement.
```

## Notes
- Using [[ChatGPT]]
### Recap
- [[ChatGPT]] in testing
- Example - testing an ecommerce application
- ![[Pasted image 20230413090647.png]]
- Saw concrete examples of [[ChatGPT]] helping with each of these
- [[ChatGPT]] was able to come up with a good testing strategy
- Was also able to come up with some decent (but not great) code to perform the tests
	- Including variations on these tests
- [[ChatGPT]] in programming
	- Able to come up with pretty good solutions (along with tests)
	- Also able to refactoring: given a link to code in [[Github]], refactor it (although wasn't verified)
	- Gave it some complex code and it was able to refactor it
- Debugging
	- Was able to both fix the code and explain the error
- Looked at some tools in the [[Artifical Intelligence]] space to help with this
	- Mostly from [[Applitools]]
- Summary: the tools are great, but have some limitations. Human mind and experience are still the essential ingredients in the Software Development Life Cycle (SDLC)

### What's Changed?
- [[ChatGPT]] on Laptop, phone, and Raspberry PI
- [[Google Bard]] was released (although lots of limitations)
	- Only English
	- Can't carry on a conversation (no context)
	- Coding support doesn't exist
	- Big fiasco: factual error in first demo :(
		- Lost $100bn in market cap
- [[ChatGPT 4]] was released
	- Accepts text and image inputs
	- "Write unit tests for the above code": did pretty well
	- "Explain this image:", then a link to an image in the blog
		- Again, does a pretty good job (image was about the test automation pyramid)
	- Give as much context as possible and be as specific as possible in the prompt
	- Experimental plugins for [[ChatGPT]]
		- You'll already find hundreds of these
	- New products have started to leverage [[ChatGPT 4]] (e.g. Microsoft Teams, &c.)
- Quality of the responses is inconsistent - definitely need to check!
- [[Github Copilot]]: lots of new, interesting products
- New [[Artifical Intelligence]] powered IDEs
- [[AutoGPT]]
	- Can do tonnes of stuff
	- Check it out on [[Github]]

### Challenges in Automation
- Slow and flaky test execution
- Sub-optimal, inefficient automation
- Incorrect, non-contextual tests
- Flaky tests:
	- Work sometimes but not others
	- Usual solution: just re-run it a few times and hope that it passes this time
	- ![[Pasted image 20230413093221.png]]
	- We'll focus on the UI / locator changes
- How to identify flaky tests?
	- ReportPortal.io - AI based reporting
	- You report the tests passing / failing there and it can summarize the information
- Flakiness due to locator / ui changes
	- Common solution is just to change to the new identifier
	- Not great, because you need to update this manually every time
	- One solution: Applitools Execution Cloud - apparently helps to deal with this?
		- Points the driver to the Execution Cloud driver instead
- Sub-optimal / inefficient automation
	- Using [[Github Copilot]] to generate a new test and to generate the code
	- 

### AI to the rescue - ChatGPT, Github Copilot, Self-Healing Execution Cloud

## Pitfalls of AI

### &c.

### &c.

## Thoughts / Next Steps
