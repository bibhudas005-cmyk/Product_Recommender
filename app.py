from flask import render_template,Flask,request,jsonify,Response
from prometheus_client import generate_latest,Counter
from dotenv import load_dotenv
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

load_dotenv()

REQUEST_COUNT=Counter("http_requests_total", "Total number of HTTP requests")
def create_app():
    app=Flask(__name__)

    vector_store=DataIngestor().ingest(load_existing=True)

    rag_chain=RAGChainBuilder(vector_store).build_chain()


    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get",methods=["POST"])
    def get_response():
        user_input=request.form["msg"]
        response=rag_chain.invoke(
             {"input":user_input},
             config={"configurable":{"session_id":"user_session"}}
        )["answer"]


        return response

    return app



    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(),mimetype="text/plain")

if __name__=="__main__":
    app=create_app()
    app.run(host="0.0.0.0",port=8000,debug=True)
