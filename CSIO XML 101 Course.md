# CSIO XML 101 Course
- [Course Homepage](https://education.csio.com/course/xml-standards-101)
## Accessing the XML standard
- Find the XML data page yunder 'Forms and Data Standards'
- You can download the standards
- XML validation tool
- You can get the full standards or just the revisions (changelog)
- Take a look at the Schemas in the dlownload
- use a tool like XMLlint using the schemas in the download
- Do not generate classes from the schemas (apparently it doesn't work very well)
	- Instead use the sample messages to make the classes :(
## Navigating the XML standard
- Help directory contains the documentation
- Contains the change history (word documents with the changes)
- Mapping documents:
	- CSIO standard forms -> CSIO XML
	- Forms and Data Standard > Industry Forms to get the actual form
	- Maps the fields on the form to the XML element
	- Xpath: Way of finding the element that stores the value 
		- A/B/C -> ``
```<A><B><C>contents</C></B></A>```
	- ACORD CSIO help file
		- Completely blank - first need to right click -> security -> unlock
	- The help file contains all of the stuff on the website (good for walking through and understanding individual elements)
	- Aggregate: group of elements
## Advantages of Supporting Documentation
- Other supporting document > EDI_XML mapping are to transfer between an EDI and an XML system
- CSIO_XML code list (contains all of the codes for the various coverages, &c.)
- Sample messages - sample XML files
- Also an XML validation tool on the website
	- Click, select version
	- Upload file
	- Hopefully it tells you that you're good! Otherwise it tells you that there are some errors and the line number, &c.
	- problems: standards@csio.com
	- Don't use the PDF file - it is huge, and just a compilation of the stuff that is in the regular help file
## Comprehending Aspects of the XML standard
- Back to the help file!
- Message based
	- Request and response
	- Request usu. from a broker management system to a carrier system
	- Rq is the request, Rs is the response
	- Generally the response contains the same stuff as in the request
	- Mod -> modify an existing policy
	- QuoteInq - quote inquiry
	- Pers - personal, Cmml - commercial
	- Open up the specific element in the help
		- Element - store an atomic value
		- Aggregate - store other elements / aggregates
		- Some types are controlled by CSIO, and others by outside organization (e.g. ISO)
		- The order of the sub-elements matters
		- Repeating - you can repeat the element multiple times. If that is not there then you must have only one
		- Sometimes there is a CHOICE in the construct
		- Required XOR - have to make a choice between the different fields
		- (%????;) is an XML entity - doesn't appear in the actual XML
		- You can search through the help, including wildcards, &c.
## Standards Evolution and Submitting Changes
- MR process is used to request new pieces of data
- Standards > Standards Requests -> Request a standards change 
	- And then fill out the form
- Standards groups meet the first Tuesday of every month, and then is voted on by the members
- Release cycle is generally beginning of the year and mid year (January and Juneish)
- WG pending - Working group pending - will be looked at at the next meeting
- Can subscribe to a given proposal
- Can also comment directly on the proposal