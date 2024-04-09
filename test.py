from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
