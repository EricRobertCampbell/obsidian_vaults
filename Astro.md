- Website: [astro.build](https://astro.build/)

## Tutorial - Build a Blog
- [Tutorial - Build a Blog](https://docs.astro.build/en/tutorial/0-introduction/)

### First Parts
- `npm create astro@latest`
- uses the `.astro` file extension
- [File-based routing](https://docs.astro.build/en/core-concepts/astro-pages/#file-based-routing)
- `/src/pages` -> generate pages
- `.astro` file contains a mix of [[HTML]] and other stuff: [HTML in `.astro`](https://docs.astro.build/en/core-concepts/astro-pages/#astro-pages)
- You can define variables in the code fences (like the frontmatter in a markdown file), and reference them as `{varName}` in the HTML
- Just like [[React]] (or [[JSX]]?) you can also write JavaScript, &c. expressions in the curly braces (e.g. `{skills.map(skill => <li>{skill}</li>)`)
- You can conditionally render using the `&&` syntax:
	- `{happy === true && <p>I am happy</p>`
 - Define file-specific [[CSS]] using the `style` tag in the HTML in the `.astro` file - they are scoped to the file they appear in
 - Define vars for the [[CSS]] 
   ```
   <style define:vars={{skillColour}}>
   .skill {
	   color: var(--skillColour)
   }
   </style>
   ```
- [Differences between `.astro` and JSX](https://docs.astro.build/en/core-concepts/astro-syntax/#differences-between-astro-and-jsx)

## Components
- `/src/components/` -> generate [[HTML]] but won't become pages
- Whatever you write in a `.astro` file becomes the default export from the file
- Import them in the code fences: `import Component from '../components/Component.astro'`
- [Astro Components Documentation](https://docs.astro.build/en/core-concepts/astro-components/#component-structure)
- Code in the `.astro` frontmatter (in the code fences) is run on the server, not on the client - you can include sensitive stuff there!
- use `const {title} = Astro.props` to grab props passed into the components
- You can import a javascript file inside a script tag!
- [[JavaScript]] written in the code fences is run at build time; that in a script tag is available for the client

## Layouts
- Found in `/src/layouts/`
- You can use the `<slot />` component in a layout to render child components / content
- [Using Slots](https://docs.astro.build/en/core-concepts/astro-components/#slots)
- You can also use named slots:
	- In the wrapper: `<slot name="slotName">`
	- Renders children with that as the slot attribute: `<div slot="slotName">content</div>`
- When you include a `layout` property in the frontmatter of an `.md` file, you can use the stuff in the frontmatter as props!
- When you have a layout that applies to md files, the actual contents (or rather, the HTML generated from them) is treated as the children (fit into the <slots /> in the layout)

## Astro API
- Using the Astro API to work with files
- use `Astro.glob` to access file information:
	- `const allPosts = Astro.glob('./posts/*.md')`
- You can specify the format for a `.md` file by setting the `layout` attribute in the frontmatter
```
---
layout: ../layouts/MarkdownPostLayout.astro
...
---
Actual md content
```
- [Astro.glob API](https://docs.astro.build/en/reference/api-reference/#astroglob)
- You can have a page which generates multiple paths / pages if you export the `getStaticPaths` function
	- returns a list of objects
		- Each one should have `{params: to pass to the template, props: the props that the templates receive}`
- If you need information to construct the page routes, write it **inside** `getStaticPaths()`.
- To receive information in the HTML template of a page route, write it **outside** `getStaticPaths()`.

## `index.astro` Routing Pattern
- In addition to the `src/pages/{pagename}.astro`, you can also get pages by having `src/pages/{pagename}/index.astro`
- [Static Routes in Astro](https://docs.astro.build/en/core-concepts/routing/#static-routes)

## RSS Feeds
- [[RSS]] feeds allow people to 'subscribe' to your blog
- `npm install @astrojs/rss`

## Astro Islands
- To add the ability to add an `island` of interactivity (e.g. with [[Preact]]): 
	- `npx astro add preact`
 - [Astro Integrations Guide](https://docs.astro.build/en/guides/integrations-guide/)
- To use an (e.g.) [[Preact]] component, need to add the `client:load` directive to make it a *hydrated component*
	- `<Greeting client:load messages=['Hello', 'Hi'] />`
	- Without this, the client won't get any [[JavaScript]] for that component!
- There are other `client` directives - for instance, `client:visible` only send the [[JavaScript]] when the component is visible on the page
- [Directives](https://docs.astro.build/en/reference/directives-reference/#client-directives)
- To add [[React]]: `npx astro add react`
	- [Futher information](https://docs.astro.build/en/guides/integrations-guide/react/)

## Sharing State
- They recommend using [nanostores](https://github.com/nanostores/nanostores)
	- [Additional Information](https://docs.astro.build/en/core-concepts/sharing-state/)

## Next Steps:
- [Guide to Content Collections](https://docs.astro.build/en/guides/content-collections/#migrating-from-file-based-routing)
- [Setting up TypeScript](https://docs.astro.build/en/guides/content-collections/#setting-up-typescript)
- [Markdown and MDX](https://docs.astro.build/en/guides/markdown-content/)
