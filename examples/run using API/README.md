# Running usecases with API 

## Motivation: Chaining of usecases
Running any use case with an API allows the user to run any usecase from any other usecase. Therefore, allowing chaining of usecases.

Aggrag provides a REST API that allows you to run your configured flows programmatically. This is particularly useful when you want to integrate Aggrag's capabilities into your own applications and/or you want to chain together multiple use cases.

## Prerequisites

Before using the API:
1. Ensure you have configured your usecase flow using the Aggrag UI
2. Mark the edges you want to execute by coloring them (only colored edges will be executed in API mode)
3. Save your flow configuration

## API Endpoint

The main endpoint for running flows is:

```bash
POST http://localhost:8000/app/run
```


## Request Format

The request should be a POST request with the following structure:

```bash
{
"flow_path": "configurations/<usecase_name>/<iteration_name>/flow-<timestamp>.cforge",
"vars": {
"@variable_name": "value",
"@another_variable": "another_value"
}
```

Where:
- `flow_path`: Path to your saved flow configuration
- `vars`: Object containing variables needed by your flow. The variables need to begin with `@`. For example, `@user_input_for_api` is a valid variable that can be sent in API request. On the UI, however, you will need to set it up as `{@user_input_for_api}`. 

## Example Usage

Here's a complete example of how to call the API:

- Copy over the `chitchat api example__1734006659822` usecase in your configurations directory in the project root directory.
- Ensure aggrag servers are running using `python -m library.app serve` and `npm run start`.
- Run the following curl request:

```bash
curl -X POST http://localhost:8000/app/run \
-H "Content-Type: application/json" \
-d '{                                               
  "flow_path": "configurations/chitchat api example__1734006659822/iteration 1/flow-1729434961134.cforge",     
  "vars": {                            
    "@user_input": "HI?",
    "@user_input 2": "I just saw your latest post and loved it!",
    "@user input 3": "Can you share some tips?"
  }                          
}'
```

## Colored vs Regular Edges

To help visualize the difference between colored and regular edges, refer to the example image:

![Colored vs Regular Edges](colored%20edge%20vs%20regular%20edge.png)

In the image above:
- **Colored Edge**: The highlighted edge that will be executed in API mode
- **Regular Edge**: The default gray edge that will be skipped during API execution

The text node connecting the colored edge has 2 API input variables that are enabled `{@user_input 2}` and `{@user_input 3}`. Those variables that are disabled won't be run even if they are included in the API request. For example, `{user_input}` in the image above. 

Valid `vars` object in the API request is:
```bash
'
"vars": {                            
    "@user_input": "HI?",
    "@user_input 2": "I just saw your latest post and loved it!",
    "@user input 3": "Can you share some tips?"
}
'       
```

## Important Notes

1. **Colored Edges**: Only edges that are marked as colored in your flow will be executed when running through the API. To color an edge:
   - Open your flow in the UI
   - Select the edge connecting two nodes
   - Save the flow

2. **Variables**: Make sure to provide all required variables that your flow expects. These are typically inputs connected to text field nodes or other input nodes in your flow. Only those variables that start with `@`, following the pattern `{@sample}` can be sent in the API request. 

3. **Flow Path**: The flow path should point to a valid .cforge file in your configurations directory. The path structure follows the pattern: `configurations/<usecase_name>/<iteration_name>/flow-<timestamp>.cforge`


## Limitations

1. Currently, the API execution is an experimental feature. 
2. Real-time visualization features available in the UI are not available through the API
3. Some node types might behave differently in API mode compared to UI mode
4. Split Node Output Format Issue:
   - When split node output connects to a prompt node, template shows "[object Object]"
   - Workaround in progress
5. API Validation (Feature not a bug):
   - Pre-execution validation for node configuration recommended
   - Generate placeholder values for required inputs
6. For Join Node, the UI execution does not save the the join 'format' in the cforge file. SO API execution, currently, only executes the default 'format' equivalent to 'JoinFormat.NumList' (makes it into numbers) 


## Supported Nodes with API
Currently the following nodes are supported:
 - Text Fields Node
 - Prompt Node
 - Split Node
 - Join Node
 - File Fields Node
 - Javascript Processor Node
 - Javascript Evaluator Node
 

Support for other nodes will be added based on the realised usefulness of the API. Should you need other nodes to be supported, kindly raise an issue or reach out! 

## Best Practices

1. Always test your flow in the UI before using it via the API
2. Ensure all required edges are properly colored
3. Keep your flow configurations organized in well-named directories
4. Document the required variables for each flow
5. Consider implementing proper error handling in your application

For more examples and use cases, check the `/examples` directory in the repository.

For more details on implementation, see the xxxPlan.md files in `library/react-server/src/services/plans/`:

curl -X POST http://localhost:8000/app/run \
      -H "Content-Type: application/json" \
      -d '{
        "flow_path": "configurations/file upload api test__1734006659822/iteration 1/flow-1736246998814.cforge",
        "vars": {
    "@user_dm": "\nThis effectively breaks up the text, providing a flow to the information being conveyed.\n\nThere you have it! This response has incorporated a list of items, code, several terminal commands, new lines, and paragraphs. I hope you find it helpful!\n"
}
      }'