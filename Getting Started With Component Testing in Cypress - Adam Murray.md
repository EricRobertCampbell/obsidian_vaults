# Getting Started With Component Testing in Cypress
- [[Adam Murray]]
- Roughly new in [[cypress]] V10
- New UI, with two different testing types: e2e and component
- spec files have been renamed "whatever.spec.ts" -> "whatever.cy.ts"
- May need to add the "cypress" types in typescript config
- Uses the `cy.mount(<Component />)` to actually mount the component
- `cy.spy`
- Can change the mount command (`component.ts`) to use a Redux wrapper, &c. - analogous to our `mountWithProviders`
```javascript
cy.intercept('GET', '/plants/', {fixture: '../plants.json'})
```
- [learn.cypress.io](learn.cypress.io)
