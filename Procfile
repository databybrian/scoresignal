worker: python main.py
webhook: gunicorn bot.webhook_handler:app --bind 0.0.0.0:$PORT --workers 2
dashboard: streamlit run streamlit_app/app.py --server.address=0.0.0.0 --server.port=$PORT







