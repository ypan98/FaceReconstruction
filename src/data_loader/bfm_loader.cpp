#ifndef BFM_LOADER_H
#define BFM_LOADER_H


class BFMLoader {
public:
	/*
	 * @Definition: Constructor
	 * 
	 * @Parameters:
	 *		- model_path: path of the H5 file which stores the Basel Face Model
	 * 
	 * @Usage: BFMLoader(model_path)
	 */
	BFMLoader(std::string model_path);

	/*
	 * @Definition: Loads the internal structure of the selected H5 file, which should contains the following information:
	 *		- Color Mean
	 *		- Coler PCA Basis
	 *		- Color PCA Variance
	 *		- Expression Mean
	 *		- Expression PCA Basis
	 *		- Expression PCA Variance
	 *		- Shape Mean
	 *		- Shape PCA Basis
	 *		- Shape PCA Variance
	 * 
	 * @Parameters: None
	 * 
	 * @Usage: BFMLoader.load()
	 */
	void load();

	/*************************************************************************************************************/
	/******************************************** Get Functions **************************************************/
	/*************************************************************************************************************/

	double 
};
#endif
