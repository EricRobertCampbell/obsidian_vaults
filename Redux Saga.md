- [Homepage](https://redux-saga.js.org/)
- Handles side effects for [[Redux]]
- Makes use of [[JavaScript Generators]]

## Useful functions:
These should all be `yield`ed - e.g. `const value = yield select(...)`
- `select(state => thing you want from state)` - make use of existing state within the saga
- `call(function, args)` - call the function, yielding the result
- `put(action)` - put the action in the queue to be processed - like dispatching it
- `takeEvery(actionType, sagaToHandleIt, ...args)` - spin up a new process to handle every occurrence of `actionType` 
