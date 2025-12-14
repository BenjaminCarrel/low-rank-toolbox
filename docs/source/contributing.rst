Contributing Guide
==================

Thank you for your interest in contributing to the Low-Rank Toolbox!

How to Contribute
-----------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/low-rank-toolbox.git
      cd low-rank-toolbox

3. **Create a development environment:**

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate low-rank-dev
      pip install -e .

4. **Create a new branch** for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

5. **Make your changes** and ensure:

   - Code follows existing style conventions
   - All tests pass: ``pytest``
   - New features include tests
   - Documentation is updated if needed

6. **Run tests locally:**

   .. code-block:: bash

      pytest -v

7. **Build documentation** to verify your changes:

   .. code-block:: bash

      cd docs
      make html
      # Open build/html/index.html in your browser

8. **Commit your changes** with clear, descriptive messages:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: brief description"

9. **Push to your fork:**

   .. code-block:: bash

      git push origin feature/your-feature-name

10. **Submit a Pull Request** on GitHub

Code Style Guidelines
---------------------

**Python Code:**

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Maximum line length: 127 characters
- Use meaningful variable names

**Docstrings:**

- Use NumPy-style docstrings
- Include mathematical notation where appropriate
- Provide examples for complex functions
- Reference academic papers when implementing published algorithms

**Example:**

.. code-block:: python

   def my_function(A: ndarray, k: int) -> ndarray:
       """
       Brief one-line description.
       
       More detailed explanation of what the function does,
       including mathematical notation if needed.
       
       Parameters
       ----------
       A : ndarray
           Description of parameter A
       k : int
           Description of parameter k
           
       Returns
       -------
       ndarray
           Description of return value
           
       Examples
       --------
       >>> import numpy as np
       >>> A = np.random.randn(10, 5)
       >>> result = my_function(A, 3)
       >>> result.shape
       (10, 3)
       
       References
       ----------
       .. [1] Author et al. (2023). Paper title. Journal.
       """
       # Implementation here
       pass

Testing Guidelines
------------------

**Writing Tests:**

- Place tests in the ``tests/`` directory
- Mirror the source code structure
- Name test files ``test_<module>.py``
- Name test functions ``test_<functionality>``

**Test Coverage:**

- Aim for >90% code coverage
- Test edge cases (empty arrays, single elements, etc.)
- Test error handling
- Test both real and complex data types

**Example Test:**

.. code-block:: python

   def test_my_function_basic():
       """Test basic functionality of my_function."""
       A = np.random.randn(10, 5)
       result = my_function(A, 3)
       
       assert result.shape == (10, 3)
       assert np.linalg.norm(result) > 0
       
   def test_my_function_edge_case():
       """Test edge case with empty array."""
       A = np.random.randn(10, 0)
       with pytest.raises(ValueError):
           my_function(A, 3)

Documentation Guidelines
------------------------

**Updating Documentation:**

When adding new features:

1. Update relevant docstrings
2. Add examples to ``docs/source/quickstart.rst`` if appropriate
3. Update API reference if adding new classes/functions
4. Build docs locally to verify formatting

**Building Documentation:**

.. code-block:: bash

   cd docs
   make clean
   make html
   # View: open build/html/index.html

Pull Request Process
--------------------

**Before Submitting:**

- ✅ All tests pass locally
- ✅ Code follows style guidelines
- ✅ Documentation is updated
- ✅ Commit messages are clear
- ✅ No merge conflicts with main branch

**PR Description Should Include:**

- What changes were made
- Why the changes were made
- How to test the changes
- Any breaking changes
- References to related issues

**Review Process:**

1. Automated checks will run (tests, linting, docs build)
2. Maintainers will review your code
3. Address any requested changes
4. Once approved, your PR will be merged!

Reporting Issues
----------------

**Bug Reports:**

Include:

- Python version
- Operating system
- Minimal reproducible example
- Expected vs. actual behavior
- Full error traceback

**Feature Requests:**

Include:

- Use case description
- Proposed API (if applicable)
- References to papers/algorithms (if applicable)
- Why this would be valuable to the community

Communication
-------------

- **GitHub Issues:** Bug reports, feature requests
- **Pull Requests:** Code contributions
- **Discussions:** General questions, ideas

Code of Conduct
---------------

Be respectful and constructive in all interactions. We welcome contributors
from all backgrounds and experience levels.

Questions?
----------

If you're unsure about anything, feel free to ask! Open an issue or
start a discussion on GitHub.

Thank you for contributing! 🎉
