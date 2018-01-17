## INSTALL SPHINX 

1. install Sphinx:

```pip install Sphinx```

2. sphinx-quickstart

Next we want to set up the source directory for the documetnation. First cd to 
root of project directory and enter:

```sphinx-quickstart```

You'll be prompted to enter a number of user options. For most you can just 
accept the defaults, but you'll want to change the following:

* Enter root path for documentation: `./docs`
* `autodoc`: y (allows automatic parsing of docstrings)
* `viewcode`: y (links documentation to source code)
* `mathjax`: y (allows mathjax in documentation)
* `githubpages`: y (allows integration with github)

## SETUP CONF.PY FILE
In conf.py file, 

1. uncomment the following lines at the top so that the conf file (located in 
./docs) can find the project in the root file:

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
```

2. change the theme to something nicer than default
html_theme = 'sphinx_rtd_theme' (personal preference) or 'classic'

3. add `sphinx.ext.napoleon` to extensions to allow parsing of google-style 
docstrings

4. to include documentation for __init__ functions, add the following functions:
(thanks to https://stackoverflow.com/a/5599712)

`def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip)`

5. To include inherited attributes and methods in documentation, add
`:inherited-members:` to modules in /docs/source/*.rst files

Example:
`
NDN\.network module
-------------------

.. automodule:: NDN.network
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
`

6. from the docs/ directory, run (for some reason this is necessary to get 
autodocs working)
`sphinx-apidoc -o source/ ../` 

7. build the documentation by running
`make html`

find landing page at
/docs/_build/html/index.html

## INTEGRATE DOCS WITH GITHUB
1. create account with readthedocs.org

2. 
