.. highlight:: shell

============
Installation
============


Stable release
--------------

To install melusine, run this command in your terminal:

.. code-block:: console

    $ pip install melusine

This is the preferred method to install melusine, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Optional dependencies
---------------------
When running Melusine in production, users may want to limit the number of packages installed.
For this purpose, Melusine makes use of optional dependencies.
The command `pip install melusine` installs only the mandatory dependencies.
Optional dependencies can be install as follows:
  * `pip install melusine[viz]` : Installs plotly and streamlit for visualization purposes
  * `pip install melusine[exchange]` : Installs exchangelib to connect Melusine with an Outlook Exchange mailbox
  * `pip install melusine[transformers]` : Installs transformers to train BERT-like models
  * `pip install melusine[all]` : Installs all the dependencies


From sources
------------

The sources for melusine can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/MAIF/melusine

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/MAIF/melusine

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/MAIF/melusine
.. _tarball: https://github.com/MAIF/melusine/tarball/master
