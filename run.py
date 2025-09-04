from app import create_app

app = create_app()

if __name__ == '__main__':
    # Respetar configuración de entorno/Config para DEBUG y host/port
    debug = app.config.get('DEBUG', False)
    host = app.config.get('HOST', '127.0.0.1')
    port = int(app.config.get('PORT', 5000))
    app.run(debug=debug, host=host, port=port)
