import requests


class SPARQLClient:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def query(self, query):
        """
        Execute a SPARQL query and return the results as a dictionary.
        """
        # Set the HTTP headers
        headers = {"Accept": "application/sparql-results+json"}

        # Encode the query for use in a URL
        encoded_query = requests.utils.quote(query)

        # Send the GET request to the SPARQL endpoint
        response = requests.get(self.endpoint_url, params={"query": query}, headers=headers)

        # Check the status code of the response
        if response.status_code != 200:
            raise Exception("SPARQL endpoint returned status code {}".format(response.status_code))

        # Return the response as a dictionary
        return response.json()



# Create a SPARQL client
client = SPARQLClient("http://localhost:8890/sparql")

# Set the SPARQL query
query = """
SELECT * WHERE {
  ?s ?p ?o
} LIMIT 10
"""

# Execute the query and print the results
results = client.query(query)
print(results)
