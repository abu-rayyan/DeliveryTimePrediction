import configparser

from webapp import app


app.run(host='0.0.0.0', port=9000, threaded=True, debug=True)
#if __name__ == '__main__':
#	app.run()
