from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import webbrowser
import threading
import time
import os


app = Flask(__name__, template_folder='templates')
CORS(app)


@app.route('/')
def home():
    return render_template('pages/index.html')  # Serve the HTML via HTTP


@app.route('/protein')
def protein():
    return render_template('pages/protein.html', proteins=project.proteins)


@app.route('/ligand')
def ligand():
    return render_template('pages/ligand.html', infos=project.ligands)


@app.route('/uploadLigand', methods=['POST'])
def uploadLigand():
    try:
        ligand_name = request.form.get('name')
        ligand_sdf = request.files['file']
        assert ligand_name not in project.ligands, f"Ligand {ligand_name} already exists!"
        fpath = str(project.upload_dir / ligand_sdf.filename)
        print(fpath)
        ligand_sdf.save(fpath)
        project.add_ligand(fpath, name=ligand_name, parametrize=False)
        res = {"status": 200, 'message': f"Ligand {ligand_name} added successfully!"}
    except Exception as err:
        res = {"status": 404, 'message': str(err)}
    return jsonify(res)


@app.route('/uploadProtein', methods=['POST'])
def uploadProtein():
    try:
        name = request.form.get('name')
        pdb = request.files['file']
        print(name)
        fpath = str(project.upload_dir / pdb.filename)
        pdb.save(fpath)
        project.add_protein(fpath, name=name)
        res = {"status": 200, 'message': f"Protein {name} added successfully!"}
    except Exception as err:
        res = {"status": 404, 'message': str(err)}
    return jsonify(res)


@app.route('/files/proteins/<path:name>', methods=['GET'])
def serve_pdb(name: str):
    return send_from_directory(
        project.proteins_dir, f'{name}/{name}.pdb'
    )

@app.route('/files/ligands/png/<path:name>', methods=['GET'])
def serve_ligand_png(protein_name: str, ligand_name: str):
    return send_from_directory(
        project.ligands_dir, f'{protein_name}/{ligand_name}.png'
    )


if __name__ == '__main__':
    import argparse
    from ..amber import AmberRbfeProject

    parser = argparse.ArgumentParser(description='EasyBFE Web-GUI')
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        required=True,
        help='Directory to a RBFE project'
    )
    parser.add_argument(
        '--init',
        dest='init',
        action='store_true',
        help='Toogle to initialize the project directory'
    )
    parser.add_argument(
        '-p', '--port',
        dest='port',
        default=5000,
        help='Port to start the Web GUI'
    )
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        help='Toogle to start debug mode'
    )
    args = parser.parse_args()


    if not args.debug:

        def openbrowser():
            webbrowser.open_new(f'http://127.0.0.1:{args.port}/')
            
        threading.Timer(1, openbrowser).start()

    global project
    project = AmberRbfeProject(wdir=args.directory, init=args.init)
    app.run(debug=True, port=args.port, host='127.0.0.1')

