from app import app, db

if __name__ == '__main__':
    with app.app_context():
        # This ensures tables are created if they don't exist
        # based on the models defined.
        db.create_all()
        # You might want to call your create_initial_admin() or load_known_faces()
        # here as well if they are not called within app.py's app context block
        # For example:
        # from app import create_initial_admin, load_known_faces
        # create_initial_admin()
        # load_known_faces()

    # The host='0.0.0.0' makes the server accessible from other devices on the network.
    # Use debug=True only for development.
    app.run(host='0.0.0.0', port=5000, debug=True)
