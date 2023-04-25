# React Router
- [Homepage](https://reactrouter.com/en/main/start/tutorial)

## Tutorial
- Create the main router in `main.tsx`
	- Create the router using `createBrowserRouter`
	- Then use it with `<RouterProvider router={router}>`
	```tsx
	1 import React from "react";
    2 import ReactDOM from "react-dom/client";
    3 import "./index.css";
    4
    5 import { createBrowserRouter, RouterProvider } from "react-router-dom";
    6
    7 const router = createBrowserRouter([
    8   { path: "/", element: <div>Hello World!</div> },
    9 ]);
   10 ReactDOM.createRoot(document.getElementById("root")).render(
   11   <React.StrictMode>
   12     <RouterProvider router={router} />
   13   </React.StrictMode>
   14 );
	```
- When the app throws an error when navigating, loading, or performing mutations, it will load an error screen.
	- You can set a custom one by setting it as the `errorElement` for that route in the router
	```jsx
	const router = createBrowserRouter([
      { path: "/", element: <Root />, errorElement: <ErrorPage /> },
    ]);
	```
	- Access the error by using `useRouteError`
	```jsx
	 import { useRouteError } from "react-router-dom";
    
     export default function ErrorPage() {
       const error = useRouteError();
       console.error(error);
    
       return (
         <div id="error-page">
           <h1>Oops!</h1>
          <p>Sorry, an unexpected error occurred.</p>
          <p>
            <em>{error.statusText || error.message}</em>
          </p>
        </div>
      );
    }
	```
- You can render a route inside another one by making it a child route in the router
 ```jsx
const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    children: [
      {
        path: "contacts/:contactId",
        element: <Contact />,
      },
    ],
  },
]);
```
- Control where the children are displayed using `<Outlet />` in the parent route
- Tight coupling between the route, the component, and some data fetched about that route
- have a `loader` that gets data (or whatever) when you navigate to a route by passing it as the `loader` parameter in the route:
```jsx
const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    loader: rootLoader,
    children: [
      {
        path: "contacts/:contactId",
        element: <Contact />,
      },
    ],
  },
]);
```
- Then in the component you can use the `useLoaderData` to get whatever data is grabbed from the `loader` function:
```jsx
import {
  Outlet,
  Link,
  useLoaderData,
} from "react-router-dom";
import { getContacts } from "../contacts";

/* other code */

export default function Root() {
  const { contacts } = useLoaderData();
  return (
    <>
      <div id="sidebar">
        <h1>React Router Contacts</h1>
        ...
```
- Form submission, &c. is done in the HTML way (where submitting a form triggers a navigation and changes the method from `POST` to `GET`, &c.)
- Difference is that no actual navigation occurs - instead, the form data is sent to a route `action` which handles the data.
- Create an `action` function which will take the URL parameters (e.g. `/contacts/:contactId`  - the action will get `params.contactId`)
- Set the `action` as an attribute to the route:
```jsx
const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    loader: rootLoader,
    action: rootAction,
    children: [
      {
        path: "contacts/:contactId",
        element: <Contact />,
      },
    ],
  },
]);
```
- The action for a route will be called if you have a `Form` (imported from `react-router-dom`) that is submittedk

- The params for a route are passed into the loader for a route, so you can access them there:
```jsx
export async function loader({ params }) {
  const contact = await getContact(params.contactId);
  return { contact };
}
```
- Actions related to a `Form` also receive a `Request` object (`request`) which contains the `FormData` (this is just a regular web thing, not [[React Router]] specific)
	- They can also return a `Response` which again emulates the usual web platform ideas (e.g. a redirect, &c.)
	- [More info](https://reactrouter.com/en/main/route/loader#returning-responses)
- Can use `NavLink` and pass a function into `className` to indicate where a user is - `isActive` and `isPending` 
- We can use the `useNavigation` hook to add a global pending UI
- `<Form action="destroy" ...>` - the action tells us where to send the action.
	- Note: you don't need to have an `element:` attribute in the router - you can Oth
- When an error occurs, it bubles up until it encounters the first `errorElement`. This means that having error components on children does not block interaction with the rest of the page (which is presumably working correctly) when an error occurs.
- When a route has children and you're at the root path, there are no children to display $\to$ need an *index path*
	- Think of this as the 'default child' to render
```jsx
const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    loader: rootLoader,
    action: rootAction,
    children: [
      { index: true, element: <Index /> },
      /* existing routes */
    ],
  },
]);
```
- `useNavigate` to imperatively navigate
	- `navigate(-1)` : navigate to the previous location (basically - like the back button)
- The default method for a `form` is `GET`
	- Then the value of the form inputs are put into the `URLSearchParams` of the `GET` request
- For the client-side routing, you can grab any URL params from the `{request}` passed into the *loader*
	- NB since this is a `GET`, not `POST`, the logic for filtering on the search params is done in the `loader`, not the `action`
- You might need to ensure that the state and the URL are in sync, espec. with regards to the search params
- Can force a submit by using `useSubmit`, then in the input `onChange` (e.g.)
```jsx
const submit = useSubmit()
<input onChange={e => submit(e.currentTarget.value)}
```
- For better loading states (&c.), the `navigation.location` will show up when you are navigating to a new URL / loading the data for it. It is gone when there is no navigation in progress
- use the `useFetcher` hook to trigger loaders and actions without causing a navigation
	- Use when e.g. you want a mutation without a navigation
	```jsx
		const fetcher = useFetcher()

		<fetcher.Form method="post"> ... </fetcher.Form>
	```
	- The formData is available on the `fetcher`, so you can use it to perform optimistic UI updates
	```jsx
	if (fetcher.formData) {
		favorite = fetcher.formData.get('favorite') === 'true'
	}
	```
	- Here if the update eventually fails, then the UI will revert to the correct value.
- If you have an expected error route in a loader, then you can `throw` a response. This will be caught by the nearest error component, and can be used to deliver more helpful error messages.
```jsx
export async function loader({ params }) {
  const contact = await getContact(params.contactId);
  if (!contact) {
    throw new Response("", {
      status: 404,
      statusText: "Not Found",
    });
  }
  return { contact };
}
```
- You can also have routes with no paths - that way they can participate in layout.
	- e.g. 
	```jsx
		createBrowserRouter([
		  {
		    path: "/",
		    element: <Root />,
		    loader: rootLoader,
		    action: rootAction,
		    errorElement: <ErrorPage />,
		    children: [
		      {
		        errorElement: <ErrorPage />,
		        children: [
		          { index: true, element: <Index /> },
		          {
		            path: "contacts/:contactId",
		            element: <Contact />,
		            loader: contactLoader,
		            action: contactAction,
		          },
		          /* the rest of the routes */
		        ],
		      },
		    ],
		  },
		]);
	```
			- here all of the children are wrapped in an error component route, meaning that any error will be caught there.
- Instead of using a route object (as we've done throughout the tutorial), we can also use `createRouteFromElements`