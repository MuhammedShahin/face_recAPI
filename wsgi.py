from app import app

if __name__ == "__main__":
    # To make the api accessible for all devices in LAN
    app.run(host='0.0.0.0')
    # app.run()
