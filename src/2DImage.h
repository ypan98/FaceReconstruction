//store data of the 2D image corresponding to the 3D face model
class 2DImage{

public:
    // get current color data
	BYTE* GetColorRGBX()
	{
		return colorFrame;
	}
	// get current depth data
	float* GetDepth()
	{
		return depthFrame;
	}


private:

    unsigned int ImageWidth;
    unsigned int ImageHeight;
    // depth info
	float* depthFrame;
    //color info
	BYTE* colorFrame;
    //current image index
    unsigned int currentIdx;
    
}
