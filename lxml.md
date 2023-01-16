# lxml
[Homepage (tutorial)](https://lxml.de/tutorial.html)

XML parsing library

## Concepts
### Attributes
- To add an attribute:
  ```python
elem.attrib['new-attrib'] = 'some value'
  ```

### Namespaces

```python
>>> etree.register_namespace("x", "http://input1.com/")
>>> input = "http://input1.com/"
>>> tag = etree.QName(input, "test")
>>> element = etree.Element(tag)
>>> element
<Element {http://input1.com/}test at 0x7f9e9cc0f440>
>>> print(etree.tostring(element).decode("utf-8"))
<x:test xmlns:x="http://input1.com/"/>
```

### Getting the XML declation at the top
```python
>>> root = etree.XML('<root><a><b/></a></root>')

>>> etree.tostring(root)
b'<root><a><b/></a></root>'

>>> print(etree.tostring(root, xml_declaration=True))
<?xml version='1.0' encoding='ASCII'?>
<root><a><b/></a></root>

>>> print(etree.tostring(root, encoding='iso-8859-1'))
<?xml version='1.0' encoding='iso-8859-1'?>
<root><a><b/></a></root>

>>> print(etree.tostring(root, pretty_print=True).decode('utf-8'))
<root>
 <a>
 <b/>
 </a>
</root>
```