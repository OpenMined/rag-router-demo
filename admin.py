from accountingSDK import AdminClient as AccountingClient


ADMIN_API_KEY = "asdf1234"


def get_client():
    """
    Get the accounting client.

    Returns
    -------
    AccountingClient
        The accounting client instance.
    """
    return AccountingClient(url="http://localhost:3000", key=ADMIN_API_KEY)


def create_user(email: str):
    """
    Create a new user in the accounting system.

    Parameters
    ----------
    email : str
        The email address of the user to create.
    """
    client = get_client()
    user, pwd = client.create_user(email=email, password="password")

    print(f"User created: {user.email} with password: {pwd}")


def add_balance(email: str, amount: float):
    """
    Add balance to a user in the accounting system.

    Parameters
    ----------
    email : str
        The email address of the user to add balance to.
    amount : float
        The amount to add to the user's balance.
    """
    client = get_client()
    user = client.add_balance(email=email, amount=amount)

    print(f"User {user.email} balance updated to {user.balance}")


def list_users():
    """
    List all users in the accounting system.
    """
    client = get_client()
    users = client.get_all_users()

    for user in users:
        print(f"User: {user.email}, Balance: {user.balance}")


if __name__ == "__main__":
    # Example usage

    # user_email = "shubham@openmined.org"
    # create_user(user_email)
    # add_balance(user_email, 100.0)

    # publisher_email = "alice@email.com"
    # create_user(publisher_email)
    list_users()
