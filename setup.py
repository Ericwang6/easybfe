import os
import setuptools
import datetime


pwd = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(pwd, 'README.md')) as f:
    long_description = f.read()


def setup(scm=None):
    setuptools.setup(
        name='easybfe',
        use_scm_version=scm,
        setup_requires=["setuptools_scm"],
        python_requires=">=3.8",
        author='Yingze (Eric) Wang',
        author_email='ericwangyz@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/Ericwang6/easybfe',
        },
        description="An open-source software to prepare and analyze relative binding free energy calculations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Drug Discovery', 'Molecular Dynamics',
            'Free Energy Calculation', 'FEP'
        ],
        license='MIT',
        packages=setuptools.find_packages(exclude=["tests", "examples", "docs"]),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
        ],
        zip_safe=False,
        entry_points={
            "console_scripts": [
                "easybfe = easybfe.cli:main",
                "easybfe-gui = easybfe.webgui:main"
            ]
        }
    )


today = datetime.date.today().strftime("%b-%d-%Y")
with open("easybfe/_date.py", 'w') as fp:
    fp.write(f'date = "{today}"')

try:
    setup(scm={"write_to": f"easybfe/_version.py"})
except:
    setup(scm=None)