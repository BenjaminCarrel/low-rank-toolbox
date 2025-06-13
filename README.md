# **Low-Rank Toolbox**

low-rank-toolbox is a Python library providing efficient data structures and algorithms for numerical linear algebra with low-rank matrices and tensors. The package is designed for researchers and practitioners who need performant and memory-conscious computations. The importable package name is lowrank.

## **Features**

* Memory-efficient storage for low-rank matrices and tensors.  
* Implementations of fundamental low-rank operations and factorizations.  
* Sub-packages for specialized algorithms like cssp (Column Subset Selection).  
* Built on top of numpy and scipy for a familiar and powerful foundation.

## **Installation**

There are two ways to install the package, depending on your needs.

### **1\. For Users (Stable Installation)**

This method is for those who want to use the lowrank package in their own projects.  
Since this is a private repository, you will need a **GitHub Personal Access Token (PAT)** to install the package.

1. **Generate a PAT:** Follow the GitHub documentation to [create a Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic). Your token must have the repo scope to access private repositories. Copy the token immediately as you will not be able to see it again.  
2. **Install with pip:** Run the following command in your terminal, replacing \<YOUR\_PAT\>, \<your-username\>, and \<branch-name\> accordingly. The branch can be main or a specific version tag (e.g., v0.1.0).  
   ```
   pip install git+https://<YOUR_PAT\>@github.com/<your-username\>/low-rank-toolbox.git@\<branch-name\>
   ```

### **2\. For Developers (Editable Installation)**

This method is for those who want to contribute to the development of low-rank-toolbox, add new features, or fix bugs.

1. **Clone the Repository:**  
   ```
   git clone https://github.com/BenjaminCarrel/low-rank-toolbox.git  
   cd low-rank-toolbox
   ```

2. **Create the Conda Environment:** This will install all the necessary dependencies, including development tools like pytest. The environment is defined in the environment.yml file.  
   ```
   conda env create \-f environment.yml  
   conda activate low-rank-dev
   ```

3. **Install in Editable Mode:** This command uses pip to link the installed package directly to your source code. Any changes you make in the src/ directory will be immediately available in your environment without needing to reinstall.  
   ```
   pip install \-e .
   ```

## **Verifying the Installation**

After installing the package using the developer instructions, you can verify that everything is set up correctly by running the test suite.  
From the root directory of the project (low-rank-toolbox/), run:  
pytest

If all tests pass, your installation is successful and the development environment is ready.

## **Quick Start**

Here is a simple example of how to use the lowrank package:  
```
#TODO: add an example
```

## **License**

This project is licensed under the MIT License. See the [LICENSE](http://docs.google.com/LICENSE) file for details.