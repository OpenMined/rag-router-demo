from loguru import logger
from syft_core import Client
from syft_llm_router.error import RouterError
from syft_llm_router.schema import RetrievalResponse
from syft_rpc import rpc
from accountingSDK import UserClient as AccountingClient

APP_NAME = "llm/rag-app"

ACCOUNTING_URL = "http://localhost:3000"


def get_acc_client():
    return AccountingClient(url=ACCOUNTING_URL, email=client.email, password="password")


def test_retrieval(client: Client, datasite: str, query: str):
    """
    Send a retrieval request to an LLM through the Syft LLM Router.

    Parameters
    ----------
    client : Client
        The Syft client instance.
    datasite : str
        The datasite to send the request to.
    query : str
        The query to send to the model.
    """

    # Send the request
    acc_client = get_acc_client()
    # datasite is the email of the recipient e.g. the publisher in this case
    transac_token = acc_client.create_transaction_token(recipientEmail=datasite)

    # Create request
    request_data = {
        "query": query,
        # "options": {"limit": 3},
    }

    url = rpc.make_url(datasite=datasite, app_name=APP_NAME, endpoint="retrieve")

    print(f"Sending retrieval request to {url} with data: {request_data}")

    try:
        future = rpc.send(
            client=client,
            url=url,
            body=request_data,
            expiry="5m",
            cache=True,
            headers={
                "X-Transaction-Token": transac_token,
            },
        )
        response = future.wait()
        response.raise_for_status()
        try:
            retrieval_response = response.model(RetrievalResponse)
            return retrieval_response.results
        except Exception:
            return response.model(RouterError)
    except Exception as e:
        logger.error(f"Error sending retrieval request: {e}")
        return None


# Example usage
if __name__ == "__main__":
    import time

    client = Client.load()

    # This is the datasite where the LLM Routing service is running
    datasite = "alice@email.com"

    start = time.time()
    print("Test retrieval....")
    query = "Provide me documents on the topic of AI?"
    response = test_retrieval(
        client=client,
        datasite=datasite,
        query=query,
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    print("Query:", query)
    print("\nTotal results: ", len(response))
    print("Sources:")
    for i, doc in enumerate(response):
        print(f"\nDocument {i + 1}: Score - {doc.score}")
        print(doc.content)
