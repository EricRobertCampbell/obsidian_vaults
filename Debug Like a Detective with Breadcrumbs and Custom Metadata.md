---
title: "Debug Like a Detective with Breadcrumbs and Custom Metadata"
authors: Andrew Rushworth
date: 2023-04-27
series: BugSnag Webinar Series
---

# Debug Like a Detective with Breadcrumbs and Custom Metadata

**Authors**: [[Andrew Rushworth]]
**Series**: [[BugSnag Webinar Series]]
**tags**: #talk #testing

## Summary

## Abstract
```
Observability is key to developers’ ability to deploy and maintain best-in-class applications for their users. But when issues arise, it’s easy to get lost searching for the answers deep within your logs. Using BugSnag’s robust error monitoring infrastructure, you can supplement, redact, and modify breadcrumbs and metadata on the fly.

Watch our solutions engineer Andrew Rushworth integrate BugSnag with a real application and instrument clever breadcrumb and metadata uses we have seen in the field.  
In this live session, we’ll cover:

-   Custom filters and custom metadata
-   How to see data across custom dimensions
-   Utilizing search & segmentation to focus on the issues that matter most
-   Live demo and Q&A
```

## Notes
- Focusing on manipulating breadcrumbs and custom metadata using callbacks
- Breadcrumbs: events within the application leading up to a crash
	- Helps tell the story of what caused the crash
	- Helps to determine the cause
- Metadata - additional context / infroamtion provided to help debug
- Callbacks?
	- Logic which can be run at the code level at the time [[BugSnag]] captures an event
	- `onError` callback
		- Accesstot he Event objects
		- Access to the application state at creash time
		- Can enhance the event w/ additional infroamtion
		- E.g. you can remove personally identifiable information (PII)
		- Can be used to stop certain events from being sent (by returning `false`)
	- `onSend` callback
		- Triggered whenever any event is sent
		- Avoid attaching additional information to a report that does not relate to the crashed session
- Demo - banking app
	- Crashes intermittently due to a change in network connectivity earlier in the session (don't know this yet)
	- Added the `onError` callback to the `bugsnag.start()`
	- `event.addMetadata()`
	- Imagine that it happens since a network driver was updated
	- Need to track connectivity though the user session leading up to a crash
	- Leverage the `onBreadcrumb` callback - triggered every time a breadcrumb is detected (automatic or manual)
	- So now we can see the connection type on every breadcrumb
	- But wait - what if it's something else?
	- Need to use the `onError` callback
		- Can seatch through breadcrumbs for key events
		- Can materialize intersting user interactions such as 'rage clicks'
	- Go through all of the previous breadcumbs and add a 'Network connection changed' custom metadata to the error $\to$ allow you to quickly filter on issues where the network changed vs. not
	- See the result in the 'Breadcrumb Info'
	- Can also search based on the custom metadata
- Some other use cases:
	- Route errors to different projects in [[BugSnag]] - e.g. check location or last breadcrumb and change the project based on that
		- E.g. monitor the stability of the 'Shopping Cart' as a separate project
	- Filter noisy but important events (e.g. only return 10% of a certain error / event type)
	- Save crash info for VIP users
		- If a VIP user has been affected by multiple crashes - set a flag
		- Create custom messaging / compensation or alert support
		- Flag can be detected in [[BugSnag]] and drive alerting