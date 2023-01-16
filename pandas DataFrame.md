# Pandas DataFrame
- Similar ideas to an [[R DataFrame]]
- Creating a DataFrame:
	- With a dictionary: ``pd.DataFrame(data=data, index=index)` - data can be a dictionary: `{column_name: [ column data]}`, index goes along the side
	```python
	df = pd.DataFrame(
		{
			'x': [1, 2, 3],
			'y': [2, 4, 8],
			'z': [100, 200, 300]
		}
	)
	```
		- You can also control the order of the columns using the `columns=['a', 'b', 'c']` parameter and the order of the index using `index=[100, 200, 300]`
	- You can also pass in a list of dictionaries:
	```python
	df = pd.DataFrame([
	{'x': 1, 'y': 2, 'z': 100},
	{'x': 2, 'y': 4, 'z': 200},
	{'x': 3, 'y': 8, 'z': 300},
	])
	```
	- Or just a list of lists / tuples!
	```python
	df = pd.DataFrame([
		[ 1, 2, 100],
		[ 2, 4, 200],
		[ 3, 8, 300],
	])
	```
- Access a column using square brackets: `df["city"]` returns the `city` column as a Pandas [[pandas Series]] with the index intact
	- You can also use `df.city` (as long as the column name is a valid Python variable name)
- You can get the entire row using `loc[]` - `df.loc[102]` will get the entire row with index `102` as a [[pandas Series]] with indices of the column names

- Write the DataFrame `df` to file using `df.to_csv("file.csv")`
- Load it in using `pd.read_csv("file.csv")`

- Access the columns using `df.columns` and the index with `pd.index`. Both of these are array-like, so we can iterate over them, &c.

- You can access the data without the index or labels using `df.to_numpy()` or `df.values()` - both will return a 2D [[numpy]] array.

## Accessing Elements
- `.loc[rows, columns]` - accepts the labels of rows and columns and returns [[pandas Series]] or [[pandas DataFrame]], depending
	-  `df.loc[:, 'city']` gets the "city" column for all indices
	-  `df.loc[:, ['city', 'name']]` returns a [[pandas DataFrame]] with all of the indices but only the 'city' and 'name' columns
- `.iloc[]` - accepts the zero-based indices of rows / columns
- `at[]` accepts the labels of rows / columns and returns a single value
- `.iat[]` accepts the zero-based index for rows / columns and returns a single value
- Both `.loc[]` and `.iloc[]` support [[Python slicing]] and [[numpy indexing]]
- All of these accessors can also be used to set the value of data - `df.loc[0:5, "score"] = [1, 2, 3, 4, 5]` will set those values in the [[pandas DataFrame]]

## Adding and Removing Data
- Add a row with `.append()`
	- Create a [[pandas Series]] representing the new row:
	```python
	new_row = pd.Series(data=['John', 12], index=df.columns, name=17)
	df = df.append(new_row)
	```
	Note:
			- The value of `name` in the series will become the new index of the row
			- The index of the [[pandas Series]] should be the same as the columns of the [[pandas DataFrame]]
			- `.append()` returns the changed [[pandas DataFrame]]
- Remove the last row with `.drop()`
	- Note that this returns the changed [[pandas DataFrame]], not the row that was dropped!
	- You can also pass in `inplace=True` to ...do exactly what it sounds like. In that case it will return `None`
- You can add a new column with `df['new row name'] = [data1, data2, ...]`
	- You can also provide only a single value, or a part of a column, and as usual it will get replicated
- If the position of the new column is important, you can use `.insert` instead. `df.insert(loc=3, column="new row name", data=[data1, data2, ...])`
	- This will insert the column so that it is at index 3 (passed in to `loc=`)
- Delete a row using the `del` statement, just like a regular [[python Dictionary]]: `del df['bad row']`
- Alternatively, you can use `.pop('column')` to remove and return the once with the specified name

## Sorting and Filtering
- Sort with `.sort_values()` - `df.sort_values(by="column name", ascending=False)`
	- You can also pass in an `axis=` to sort the rows or the columns
	- To sort by multiple columns, just pass in the names and ascending values as lists: `df.sort_values(by=['first', 'second'], ascending=[False, True])`
		- This would sort first by the `'first'` column (descending), and then if there is a tie there, by `'second'` (ascending)

- You can also filter by passing in an array of `bool`s: `df[df['score'] > 80]`

## Missing Data
- `df.fill_na('value to fill')`
	- You can also use the `method` attribute to instead fill the value with the value directly above or below it
- You can also fill missing data by interpolation using `df.interpolate()`
- Drop rows / columns with missing data with `df.dropna()`

## Iterating
- `df.items()` - iterates over the columns
- `df.iteritems()` - iterates over the columns
- `df.iterrows()` - iterates over the rows
- `df.itertuples()` - iterate over the rows and get named tuples

## Time Series
- Generate a time series with `pd.date_range`: `dt = pd.date_range(start='2022-03-18 00:00:00.0', periods=24, frequency='H')`
- You can split into specific time periods (e.g. 4 hour blocks)
- Also you can set a rolling window of times

## Resources
- [RealPython - DataFrames](https://realpython.com/pandas-dataframe/)
- [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#hierarchical-indexing-multiindex)