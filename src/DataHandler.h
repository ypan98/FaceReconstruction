#include "FaceModel.h"

class DataHandler{
public:
    bool ReadFileList(const std::string& filename, std::vector<std::string>& result)
	{
		
    }
    bool Init(const std::string& datasetDir)
	{
		baseDir = "../data/samples";

		// read depth and rgb filename lists
		//if (!ReadFileList(baseDir + "depth" + "", DepthImages)) return false;
		//if (!ReadFileList(datasetDir + "rgb" + "", ColorImages)) return false;

		// image resolutions
		ImageWidth = 250;
		ImageHeight = 250;
	
		depthFrame = new float[ImageWidth*ImageHeight];
		for (unsigned int i = 0; i < ImageWidth*ImageHeight; ++i) depthFrame[i] = 0.5f;

		colorFrame = new BYTE[4* ImageWidth*ImageHeight];
		for (unsigned int i = 0; i < 4*ImageWidth*ImageHeight; ++i) colorFrame[i] = 255;

		currentIdx = -1;
		return true;
	}
	

private:
    // base dir
	std::string m_baseDir;
	// filenamelist depth
	std::vector<std::string> DepthImages;
	// filenamelist color
	std::vector<std::string> ColorImages;

}