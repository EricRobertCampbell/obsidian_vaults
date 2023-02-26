- Experimental methods have become more automated and more efficient
- *Experimental optimization* - determine the optimal settings to use to run a business
- Systems built by three different types of engineers:
	- Machine learning engineersa ("MLE")
	- Quantitative traders ('quants')
	- Software engineers ("SWE")
- ![[01image002.png]]
- Progressive improvement
1. Implement change:
2. Evaluate offline:
	- Business impact is measured away from the production system. 
	- Typically uses data previously logged to make estimates
	- If the preliminary estimates show a negative impact, then it's rejected
3. Measure online
	- Change is pushed to production
	- May require some configuration
	- Experimentation - try to find the optimal parameters (or combination of parameters)
- Book deals with stage 3 - running the experiment on production
- Need to minimize the *costs* associated with running an experiment
- Engineering workflows for the three different kinds of experimenter

## Examples of Engineering Workflows
### MLE Workflow
![[01image003.png]]
- Implement change: fits the new predictor to the logged data. If the results are better than the existing predictor, move to the next stage
- Deploy to production - if the results are better than the old method, then they stay; otherwise go back to the old way
### Quant workflow
![[01image004.png]]
- E.g. new trading system
- Need to optimize both return and risk
- Run new system through a simulation (backtest) on historical market data
### SWE Workflow
![[01image005.png]]
- SWE problems: don't involve building models from data
- E.g. websites, &c.
- E.g. response time of a search engine - reduce the 'bounce rate' - probability that a user will navigate away from the site after viewing just one page
- Evaluate offline: test the new engine against previous searches to time them (want to decrease the time)

## Measuring by experiment
### Experimental methods
- Focus just on the thing you're changing

### Practical problems and pitfalls
- Deviating from assumptions and errors, &c.

### Why are experiments necessary
- Several other tools to assess the business metric impact of a change (apart from the experiment)
	1. Domain knowledge
		 - Specialized knowledge of a field, market, business, &c.
		 - Not infallible - expert ideas seem to work about 10% - 50% of the time
		 - Complexity - had to reason through all of the downstream effects of a change
	2. Offline model quality
		 - Not uncommon to improve a model, yet not see a business improvement
		 - E.g. when you tested the data, it was without the new system - unanticipated response. 
		 - Maybe the users were used to the old system and don't like the new one
		 - Offline data is based on a world without the new model in it
		 - [[Online-offline gap]]
	3. Simulation
		 - Tools that measure a system's business metrics offline
		 - Can produce better fit because they use lots more data
		 - E.g. simulation can use 10 years of historical data. If you want to run an experiment with 10 years of data, you need to run it for 10 years!
		 - May be biased or inaccurate (see above point)

 