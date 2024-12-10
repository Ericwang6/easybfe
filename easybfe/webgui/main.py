from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Blueprint, current_app
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import webbrowser
import threading
import time
import os
from pathlib import Path
from typing import Dict, Any
import argparse
from .celery_app import make_celery
from ..amber import AmberRbfeProject


app = Flask(__name__, template_folder='templates', root_path=os.path.dirname(os.path.abspath(__file__)))
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery = make_celery(app)

@celery.task(name='easybfe.webgui.main.add_ligand_async')
def add_ligand_async(data: Dict[str, Any]):
    fpath = data.pop('fpath')
    proj_path = data.pop('project_path')
    AmberRbfeProject(proj_path).add_ligands(
        fpath,
        num_workers=1,
        **data
    )

def custom_url_for(endpoint, **values):
    prefix = os.getenv('ONDEMAND_PREFIX', '')
    return prefix + url_for(endpoint, **values)


@app.context_processor
def replace_url_for():
    return dict(url_for=custom_url_for)


@app.route('/')
def home():
    return render_template('pages/index.html')  # Serve the HTML via HTTP


@app.route('/protein')
def protein():
    return render_template('pages/protein.html', proteins=project.proteins)


@app.route('/ligand')
def ligand_page():
    df = project.gather_ligands_info()
    table = render_table(df, onclick="displayLigand(this)")
    return render_template('pages/ligand.html', ligand_table=table)


def render_table(df, onclick="javascript:void(0)"):
    lines = ['<table class="hover table table-bordered" id="dataTable" width="100%" cellspacing="0"><thead><tr>']
    for col in df.columns:
        lines.append(f'<th>{col}</th>')
    lines.append('</tr></thead><tbody>')
    for index, row in df.iterrows():
        lines.append(f'<tr style="cursor: pointer;" onclick="{onclick}">')
        for col in df.columns:
            if isinstance(row[col], float):
                string = f'{row[col]:.2f}'
            else:
                string = str(row[col])
            lines.append(f'<td name="{col}">{string}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table>')
    return ''.join(lines)


@app.route('/add_ligand')
def add_ligand_page():
    return render_template('pages/add_ligand.html', proteins=project.proteins)


@app.route('/perturbation')
def perturbation_page():
    df = project.gather_perturbations_info()
    table = render_table(df, onclick="displayPert(this)")
    return render_template('pages/perturbation.html', pert_table=table)


@app.route('/add_perturbation')
def add_perturbation_page():
    return render_template('pages/add_perturbation.html', proteins=project.proteins)


@app.route('/uploadLigand', methods=['POST'])
def uploadLigand():
    ligand_sdf = request.files['file']
    fpath = str(project.upload_dir / ligand_sdf.filename)
    ligand_sdf.save(fpath)
    data = {
        'project_path': str(project.wdir),
        'fpath': fpath,
        'name': request.form.get('name'),
        'protein_name': request.form.get('protein'),
        'forcefield': request.form.get('forcefield'),
        'charge_method': request.form.get('charge'),
        'parametrize': True
    }
    task = add_ligand_async.apply_async(args=[data])
    return jsonify({'status': 202, 'task_id': task.id})


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

@app.route('/files/ligands/png/<path:protein_name>/<path:ligand_name>', methods=['GET'])
def serve_ligand_png(protein_name: str, ligand_name: str):
    return send_from_directory(
        project.ligands_dir, f'{protein_name}/{ligand_name}/{ligand_name}.png'
    )

@app.route('/files/perturbations/png/<path:protein_name>/<path:pert_name>', methods=['GET'])
def serve_perturbation_mapping_png(protein_name: str, pert_name: str):
    return send_from_directory(
        project.rbfe_dir, f'{protein_name}/{pert_name}/atom_mapping.png'
    )


@app.route('/getLigands', methods=['POST'])
def getLigands():
    jdata = request.get_json()
    ligands = project.ligands.get(jdata['protein_name'], [])
    return jsonify({'ligands': ligands})


@app.route('/getPresetConfig', methods=['POST'])
def getPresetConfig():
    jdata = request.get_json()
    setting = jdata.get('setting', '16lambda_5ns')
    with open(os.path.join(os.path.dirname(__file__), f'config_{setting}.json')) as f:
        content = f.read()
    return jsonify({'content': content})


def main():
    parser = argparse.ArgumentParser(description='EasyBFE Web-GUI')
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        default=".",
        help='Directory to a RBFE project'
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
    parser.add_argument(
        '--host',
        dest='host',
        default='127.0.0.1',
        help='Host to run the Web-GUI. Swith to 0.0.0.0 if you deploy the code on a remote server.'
    )
    args = parser.parse_args()

    if not args.debug:
        def openbrowser():
            webbrowser.open_new(f'http://{args.host}:{args.port}/')
        threading.Timer(1, openbrowser).start()
    
    global project
    project = AmberRbfeProject(wdir=args.directory)
    app.config['PROJECT'] = project
    app.run(debug=args.debug, port=args.port, host=args.host)


if __name__ == '__main__':
    main()

    
