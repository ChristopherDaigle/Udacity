/* Database Normalization:

1. Are the tables storing logical groupings of the data?
2. Can I make changes in a single location, rather than in
  many tables for the same information?
3. Can I access and manipulate data quickly and efficiently? */

/* JOIN and ON */
SELECT
  orders.*
FROM
  orders
JOIN
  accounts
ON
  orders.account_id = accounts.id
LIMIT
  5;
