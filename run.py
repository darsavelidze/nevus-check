"""
Main entry point for the Nevus Analyzer web application
"""

from app import create_app

if __name__ == '__main__':
    app = create_app()
    print("=" * 50)
    print("Nevus Analyzer Web Application")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
