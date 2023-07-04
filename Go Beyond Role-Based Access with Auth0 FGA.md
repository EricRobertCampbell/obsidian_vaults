---
title: "Go Beyond Role-Based Access with Auth0 FGA"
authors: Adrian Tam
date: 2023-05-16
series: DevDay23
---

# Go Beyond Role-Based Access with Auth0 FGA

**Authors**: [[Adrian Tam]]
**Series**: [[DevDay23]]
**tags**: #talk 

## Summary

## Abstract
```

```

## Notes
- [[Authentication]] vs. [[Authorization]]
	- Authentication: "Is this person Kate?"
	- Authorization: "Is Kate allowed to reply?"
 - Fine-grained authorization:
	 - Have access to specific aobjects because the orject or its parents have been shared with them or one of the groups they belong to
	 - e.g. Google Drive
- Why an Authorization service?
	- Better auditability
	- Decoupling access policy enformcement from the policy decisions
	- Consititent APIs across teams
	- Enforcement of authorization across service or product boundaries
- Requirements:
	- Highly available
	- Low latency
	- correct
	- flexible
- One way: Role-Based Access Control
	- E.g. all "Employees" can access this resource
	- Downsides: doesn't scale well when there are many roles
- Can also use RBAC with permissions: check for 'view:document', e.g.
	- How can you solve the case where you want to share a single document?
	- How do you revoke the permission? Since it's in the JWT, you would need to wait for that to expire
- Authorization with Attribute Based Access Control (ABAC)
	- Downsides: nesting - have to recurse through the whole chain
	- Have to go to the product database and increase the load
	- Have to repeat the check for every type of resource
 - Relationship-Baed Access Control (ReBAC)
	 - ![[Pasted image 20230516111203.png]]
- What is [[Auth0]] FGA?
	- Inspired by the Google Zanzibar paper (describes how Google authorization was built)
- Seems to be ReBAC based
- Open-Source: OpenFGA
	- Auditable and extensible
- ![[Pasted image 20230516111407.png]]
- ![[Pasted image 20230516111456.png]]
- ![[Pasted image 20230516111659.png]]
- Can query the relationships:
	- ![[Pasted image 20230516111843.png]]
- ![[Pasted image 20230516111922.png]]
	- Seems way simpler, but it seems like a lot of the complexity is just being hidden in the SDK...
- ![[Pasted image 20230516112022.png]]
- ![[Pasted image 20230516112125.png]]
- 