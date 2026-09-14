from app.search.base import ShoppingListing


def average_price(listings: list[ShoppingListing]) -> float:
    return round(sum(listing.price for listing in listings) / len(listings), 2)


def best_value(listings: list[ShoppingListing]) -> ShoppingListing:
    return min(listings, key=lambda listing: listing.price)
