curl -X 'POST' \
  'http://localhost:8000/customer-intel' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "customer": {
      "customer_id": "CUST1001",
      "age": 35,
      "education": "tertiary",
      "job": "management",
      "balance": 2000.50,
      "duration": 220,
      "complaint": "Unexpected transaction fees charged."
    },
    "question": "What was their specific complaint about transaction fees?"
  }'