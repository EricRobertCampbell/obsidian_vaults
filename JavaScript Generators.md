```js
function* simpleGenerator() {
	yield 1;
	yield 2;
	yield 3;
}
```
- Create using the `function*` keyword
- `yield` is a special kind of `return`, but it doesn't stop execution
- A *generator function* returns a special `Generator` object
```js
const generatorObject = simpleGenerator()
generatorObject.next() // {value: 1, done: false}
generatorObject.next() // {value: 2, done: false}
generatorObject.next() // {value: 3, done: false}
generatorObject.next() // {value: undefined, done: true}
```
- Get the values using the `next` attribute on the generator
- `next()` returns an object with `{value: yielded value, done: true / false}`
- Generator 'pauses' at the `yield` and continues from there to the next `yield` 
- Generator objects are independent - you can have multuople going at the same time
```js
const generator1 = simpleGenerator()
const generator2 = simpleGenerator()

generator1.next() // {value: 1, done: false}
generator1.next() // {value: 2, done: false}
generator2.next() // {value: 1, done: false} - independent of the first one
```
- One use case is infinte loops that don't lock the computer:
```js
// Generates as many IDs as you want (up to the limit of the JS numerical system)
// One huge advantage - you don't need to keep track of which ids you've assigned - that happens within this function!
function* generateId() {
	let id = 1;
	while (true) {
		yield id;
		id++;
	}
}

const generatorObject = generateId()
generatorObject.next() // {value: 1, done: false}
generatorObject.next() // {value: 2, done: false}
generatorObject.next() // {value: 3, done: false}
```
- Another instance: using it as an iterator
- A [[JavaScript Iterator]] is something that has a `next()` attribute which returns a `{value: ???, done: true / false}` object
```js
// e.g. creatring an iterator from an array
function* generator(array) {
	for (let i = 0; i < array.length; i++) {
		yield array[i]
	}
}
const generatorObject = generator([1, 3, 5])
generatorObject.next() // {value: 1, done: false}
generatorObject.next() // {value: 3, done: false}
generatorObject.next() // {value: 5, done: false}
generatorObject.next() // {value: undefined, done: true}
```
- You can also pass information *back into* a generator using the `yield` keyword (interior to the generator) and passing an argument into the `next()` function (when calling the generator object). Let's see how this works with the id generator:
```js
function* generateId() {
	let id = 1;
	while (true) {
		const increment = yield id;
		if (increment !== null) {
			id += increment;
		} else {
			id += 1;
		}
		id++;
	}
}

const generatorObject = generateId()
generatorObject.next() // {value: 1, done: false}
generatorObject.next(4) // - this is passed *into* the generator at the yield statement (so increment = 4) -> {value: 5, done: false}
generatorObject.next() // {value: 6, done: false}
generatorObject.next() // {value: 7, done: false}
```
- The program flow here is a bit weird:
	- The first time, it paused at the yield (yields 1) but *doesn't get `increment`*
	- Then I passed in `4`
	- That `4` is passed into `increment`, it continues with the loop until it hits the next `yield` - yielding 4 + 1 = 5
- Note that passing an argument into `next()` on the first iteration doesn't do anything - no yield to pause at!
- You can also `return` in a generator - that immediately stops the iterator
- There's also a `.throw()` function which throws an error
```js
generatorObject.throw(new Error('Bang!'))
```

- We can also iterator over a generator object using a `for...of` loop and spread it into an array:
```js
// for...of loop
function* simpleGenerator() {
	yield 'a';
	yield 'b';
	yield 'c';
}
const obj = simpleGenerator()
for (let value of obj) {
	console.log(value)
}
// result is 'a', 'b', 'c'

// using spread syntax
const obj2 = simpleGenerator()
const newArray = [...obj2] // -> ['a', 'b', 'c']
```
- NB `return` values are ignored by `for...of` since the returned value has {`done: true`}
- On a generator there's also a `.return()` method which immediately returns that value and terminates the generator (as thought the `yield` it was paused at was a `return` statement)
- This can be useful for e.g. terminating an async process (maybe you need to cancel a sequence of API calls or something)
```js
function* simpleGenerator() {
	yield 'a';
	yield 'b';
	yield 'c';
}
const obj = simpleGenerator()
obj.next() // {value: 'a', done: false}
obj.return('Test') // {value: 'Test', done: true}
obj.next() // {value: undefined, done: true}
```
- Notice that the object returned had a *value* and *done: true*. That only happens with a `return` statement inside the generator or when using the `.return()` method
- `.throw()` allows you to pass in an exception, terminating the error and finishing the generator
```js
function* simpleGenerator() {
	try { 
		yield 'a';
		yield 'b';
		yield 'c';
	} catch (e) { 
		console.log(e)
	}
}

const obj = simpleGenerator()
obj.next() // {value: 'a', done: false}
obj.throw(new Error('BANG!')) // prints out the error, and {value: undefined, done: true}
obj.next() // {value: undefined, done: true}
```
- For whatever reason, if the `yield` is in a loop then this behaviour is different - the error is successfully caught and the generator continues...
- You can also use the `yield*` keyword to delegate yielding to another generator:
```js
function* delegate() {
	yield 3;
	yield 4;
}
function* begin() {
	yield 1;
	yield 2;
	yield* delegate()
}

const obj = begin()
for (let value of obj) {
	console.log(value)
} // 1, 2, 3, 4
```
- `yield*` can also delegate to any [[JavaScript Iterator|iterator]], like [[JavaScript Array|Array]] or [[JavaScript Map|Map]]
```js
function* generate() {
	yield* [1, 2, 3]
}
for (let val of generate()) {
	console.log(val)
} // 1, 2, 3 - *not* the array [1, 2, 3]
```
- You can see the values that you pass into the `.next()` function:
```js
function* generatorFunction() {
	console.log(yield)
	console.log(yield)

	return 'The End'
}

const generator = generatorFunction()

generator.next(100) // logs undefined, {value: undefined, done: false}
generator.next(200) // logs 200, {value: undefined, done: false}
generator.next(300) // logs 300, {value: "The End", done: true}
generator.next(400) // no log, {value: undefined, done: true}
```
- You can also seed the generator with an initial value by passing that value (or whatever) into the generator function
```js
function* generator(value) {
	while (true) {
		value = yield value * 10;
	}
}
const obj = generator(0) // initial value
for (let i = 0; i < 5; i++) {
	console.log(generator.next(i).value)
} // 0, 10, 20, 30, 40
```
- You can mimic [[JavaScript Async Await]] using generators:
```js
// async version
const asyncGetUsers = async function() {
	const response = await fetch('https://jsonplaceholder.typicode.com/users')
	const json = await resonse.json()
	
	return json
}
// call the function and log the response
asyncGetUsers.then(response => console.log(response))
// something like
[
  {
    "id": 1,
    "name": "Leanne Graham",
    "username": "Bret",
    "email": "Sincere@april.biz",
    "address": {
      "street": "Kulas Light",
      "suite": "Apt. 556",
      "city": "Gwenborough",
      "zipcode": "92998-3874",
      "geo": {
        "lat": "-37.3159",
        "lng": "81.1496"
      }
    },
    "phone": "1-770-736-8031 x56442",
    "website": "hildegard.org",
    "company": {
      "name": "Romaguera-Crona",
      "catchPhrase": "Multi-layered client-server neural-net",
      "bs": "harness real-time e-markets"
    }
  },
  {
    "id": 2,
    "name": "Ervin Howell",
    "username": "Antonette",
    "email": "Shanna@melissa.tv",
    "address": {
      "street": "Victor Plains",
      "suite": "Suite 879",
      "city": "Wisokyburgh",
      "zipcode": "90566-7771",
      "geo": {
        "lat": "-43.9509",
        "lng": "-34.4618"
      }
    },
    "phone": "010-692-6593 x09125",
    "website": "anastasia.net",
    "company": {
      "name": "Deckow-Crist",
      "catchPhrase": "Proactive didactic contingency",
      "bs": "synergize scalable supply-chains"
    }
  },
  {
    "id": 3,
    "name": "Clementine Bauch",
    "username": "Samantha",
    "email": "Nathan@yesenia.net",
    "address": {
      "street": "Douglas Extension",
      "suite": "Suite 847",
      "city": "McKenziehaven",
      "zipcode": "59590-4157",
      "geo": {
        "lat": "-68.6102",
        "lng": "-47.0653"
      }
    },
    "phone": "1-463-123-4447",
    "website": "ramiro.info",
    "company": {
      "name": "Romaguera-Jacobson",
      "catchPhrase": "Face to face bifurcated interface",
      "bs": "e-enable strategic applications"
    }
  },
  {
    "id": 4,
    "name": "Patricia Lebsack",
    "username": "Karianne",
    "email": "Julianne.OConner@kory.org",
    "address": {
      "street": "Hoeger Mall",
      "suite": "Apt. 692",
      "city": "South Elvis",
      "zipcode": "53919-4257",
      "geo": {
        "lat": "29.4572",
        "lng": "-164.2990"
      }
    },
    "phone": "493-170-9623 x156",
    "website": "kale.biz",
    "company": {
      "name": "Robel-Corkery",
      "catchPhrase": "Multi-tiered zero tolerance productivity",
      "bs": "transition cutting-edge web services"
    }
  },
  {
    "id": 5,
    "name": "Chelsey Dietrich",
    "username": "Kamren",
    "email": "Lucio_Hettinger@annie.ca",
    "address": {
      "street": "Skiles Walks",
      "suite": "Suite 351",
      "city": "Roscoeview",
      "zipcode": "33263",
      "geo": {
        "lat": "-31.8129",
        "lng": "62.5342"
      }
    },
    "phone": "(254)954-1289",
    "website": "demarco.info",
    "company": {
      "name": "Keebler LLC",
      "catchPhrase": "User-centric fault-tolerant solution",
      "bs": "revolutionize end-to-end systems"
    }
  },
  {
    "id": 6,
    "name": "Mrs. Dennis Schulist",
    "username": "Leopoldo_Corkery",
    "email": "Karley_Dach@jasper.info",
    "address": {
      "street": "Norberto Crossing",
      "suite": "Apt. 950",
      "city": "South Christy",
      "zipcode": "23505-1337",
      "geo": {
        "lat": "-71.4197",
        "lng": "71.7478"
      }
    },
    "phone": "1-477-935-8478 x6430",
    "website": "ola.org",
    "company": {
      "name": "Considine-Lockman",
      "catchPhrase": "Synchronised bottom-line interface",
      "bs": "e-enable innovative applications"
    }
  },
  {
    "id": 7,
    "name": "Kurtis Weissnat",
    "username": "Elwyn.Skiles",
    "email": "Telly.Hoeger@billy.biz",
    "address": {
      "street": "Rex Trail",
      "suite": "Suite 280",
      "city": "Howemouth",
      "zipcode": "58804-1099",
      "geo": {
        "lat": "24.8918",
        "lng": "21.8984"
      }
    },
    "phone": "210.067.6132",
    "website": "elvis.io",
    "company": {
      "name": "Johns Group",
      "catchPhrase": "Configurable multimedia task-force",
      "bs": "generate enterprise e-tailers"
    }
  },
  {
    "id": 8,
    "name": "Nicholas Runolfsdottir V",
    "username": "Maxime_Nienow",
    "email": "Sherwood@rosamond.me",
    "address": {
      "street": "Ellsworth Summit",
      "suite": "Suite 729",
      "city": "Aliyaview",
      "zipcode": "45169",
      "geo": {
        "lat": "-14.3990",
        "lng": "-120.7677"
      }
    },
    "phone": "586.493.6943 x140",
    "website": "jacynthe.com",
    "company": {
      "name": "Abernathy Group",
      "catchPhrase": "Implemented secondary concept",
      "bs": "e-enable extensible e-tailers"
    }
  },
  {
    "id": 9,
    "name": "Glenna Reichert",
    "username": "Delphine",
    "email": "Chaim_McDermott@dana.io",
    "address": {
      "street": "Dayna Park",
      "suite": "Suite 449",
      "city": "Bartholomebury",
      "zipcode": "76495-3109",
      "geo": {
        "lat": "24.6463",
        "lng": "-168.8889"
      }
    },
    "phone": "(775)976-6794 x41206",
    "website": "conrad.com",
    "company": {
      "name": "Yost and Sons",
      "catchPhrase": "Switchable contextually-based project",
      "bs": "aggregate real-time technologies"
    }
  },
  {
    "id": 10,
    "name": "Clementina DuBuque",
    "username": "Moriah.Stanton",
    "email": "Rey.Padberg@karina.biz",
    "address": {
      "street": "Kattie Turnpike",
      "suite": "Suite 198",
      "city": "Lebsackbury",
      "zipcode": "31428-2261",
      "geo": {
        "lat": "-38.2386",
        "lng": "57.2232"
      }
    },
    "phone": "024-648-3804",
    "website": "ambrose.net",
    "company": {
      "name": "Hoeger LLC",
      "catchPhrase": "Centralized empowering task-force",
      "bs": "target end-to-end models"
    }
  }
]
```
- Now let's do it with generator functions
```js
// we'll write asyncAlt later
const generatorGetUsers = asyncAlt(function*() {
	const response = yield fetch('https://jsonplacegolder.typicode.com/users')
	const json = yield response.json()

	return json
})

// asyncAlt - has a generator function as a parameter, which is our function which  yields promises which fetch returns. This will return a function which resolves every promise that it finds, until the last one
function asyncAlt(generatorFunction) {
	return function() {
		// create and assign the generator object
		const generator = generatorFunction()

		// define a function that accepts the next iteration of the generator
		function resolve(next) {
			// if the generator is closed and there are no more values to yield, resolve the last value
			if (next.done) {
				return Promise.resolve(next.value) // next.value is the promise yielded from fetch
			}

			// if there are still values to yield, they are promises and must be resolved
			return Promise.resolve(next.value).then(response => resolve(generator.next(response)))
		}

	// begin resolving promises
	return resolve(generator.next())
	}
}

generatorGetUsers().then(response => console.log(respone)) // same output as before
```
## Sources
- [MDN Docs - Generators](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Generator)
- [YouTube - WebDev Simplified - Learn JavaScript Generators in 12 minutes](https://www.youtube.com/watch?v=IJ6EgdiI_wU)
- [DigitalOcean - Understanding Generators in JavaScript](https://www.digitalocean.com/community/tutorials/understanding-generators-in-javascript)