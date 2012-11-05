#include "cuda.h"
#include "DisplayResponse.h"

#include "DisplayKernel.cu"
#include "ComplexAbsKernel.cu"

/**
* TODO: in CUDA 4.1 there is a crash caused by draw buffer. Memory leak on GPU?
* There is also a small memory leak on the CPU side.
* Change from using depricated OpenGL interoperability.
**/

// Constructor
DisplayResponse::DisplayResponse(Dimension<int> dim, float dynamRange, bool envelope) : dispDim(dim), dynamicRange(dynamRange), envelope(envelope) 
{
	pbo = 0;
}

// Destructor
DisplayResponse::~DisplayResponse() 
{
	deletePBO();
}

void DisplayResponse::drawBuffer()
{
	// display result
	// Clear the color part of both back buffers
	glClear(GL_COLOR_BUFFER_BIT);

	// Draw image from right and left PBO
	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0, 0);

	glDrawBuffer(GL_BACK);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glDrawPixels(getWidth(), getHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0); // memory bugg here? strange

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glFinish();
}

/**
* Launch CUDA kernel who maps complex result to openGL pixel buffer object (PBO).
*
* Before copying to pbo, result is normalized 
*
**/
void DisplayResponse::mapResponseToDisplay(const cuComplex* result, const uint obsW, const uint obsH, const bool resultOnGPU) 
{

	// TODO: Extraxt some of the code here out to private helper functions

	// Take abs of result
	float* abs_buffer;
	const uint n = obsW * obsH;
	cuUtilsSafeCall( cudaMalloc<float>(&abs_buffer, sizeof(float)*n) );

	cuComplex* result2;
	if (!resultOnGPU) {
		cuUtilsSafeCall( cudaMalloc<cuComplex>(&result2, obsW * obsH * sizeof(cuComplex)) );
		cuUtilsSafeCall( cudaMemcpy(result2, result, obsW * obsH * sizeof(cuComplex), cudaMemcpyHostToDevice) );
	}

	const dim3 abs_block(256, 1, 1);
	const dim3 abs_grid((n-1)/abs_block.x + 1, 1);
	ComplexAbsKernel<<<abs_grid, abs_block>>>(abs_buffer, (resultOnGPU? result : result2), n, envelope);

	//cuUtilsSafeCall( cudaThreadSynchronize() );

	// Find maximum and minimum using CUBLAS .
	//cublasInit();
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {printf("CUBLAS init failed (error %d)\n", stat); return;}

	int maxValIdx = 1;
	stat = cublasIsamax(handle, n, abs_buffer, 1, &maxValIdx);
	if (stat != CUBLAS_STATUS_SUCCESS) {printf("CUBLAS: Finding max failed (error %d)\n", stat); return;}

	int minValIdx = 1;
	stat = cublasIsamin(handle, n, abs_buffer, 1, &minValIdx);
	if (stat != CUBLAS_STATUS_SUCCESS) {printf("CUBLAS: Finding min failed (error %d)\n", stat); return;}

	maxValIdx -= 1; // CUBLAS uses Fortran indexing (1,2,3,...,n)
	minValIdx -= 1;
	
	//cublasShutdown();
	cublasDestroy(handle);

	// copy max value to constant memory (defined in DisplayKernel.cu)
#if CUDA_VERSION == 5000
   cuUtilsSafeCall( cudaMemcpyToSymbol(maxValue, &abs_buffer[maxValIdx], sizeof(float), 0, cudaMemcpyDeviceToDevice) );
   cuUtilsSafeCall( cudaMemcpyToSymbol(minValue, &abs_buffer[minValIdx], sizeof(float), 0, cudaMemcpyDeviceToDevice) );
#else
   cuUtilsSafeCall( cudaMemcpyToSymbol("maxValue", &abs_buffer[maxValIdx], sizeof(float), 0, cudaMemcpyDeviceToDevice) );
   cuUtilsSafeCall( cudaMemcpyToSymbol("minValue", &abs_buffer[minValIdx], sizeof(float), 0, cudaMemcpyDeviceToDevice) );
#endif
   
	if (!resultOnGPU) {
		cuUtilsSafeCall( cudaFree(result2) );
	}

	/* DEBUG CODE */
	/*
	float* tempAbsBuffer = (float *)malloc(sizeof(float)*n);
	cuUtilsSafeCall( cudaMemcpy(tempAbsBuffer, abs_buffer, sizeof(float)*n, cudaMemcpyDeviceToHost) );
	for (uint i = 0; i < n; i++) {
	printf("%2.2f ", tempAbsBuffer[i]);
	if (i != 0 && (i % obsH) == obsH-1)
	printf("\n");
	}
	printf("\n");
	free(tempAbsBuffer);
	*/

	/*float tempMaxValue = 0.0f;
	cuUtilsSafeCall( cudaMemcpy(&tempMaxValue, &abs_buffer[maxValIdx], sizeof(float), cudaMemcpyDeviceToHost) );
	printf("Max value: %f\n", tempMaxValue);

	float tempMinValue = 0.0f;
	cuUtilsSafeCall( cudaMemcpy(&tempMinValue, &abs_buffer[minValIdx], sizeof(float), cudaMemcpyDeviceToHost) );
	printf("Min value: %f\n", tempMinValue);
	*/
	/*------------*/


	/*if (cublasStat != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf(stderr, "CUBLAS ERROR");
	}*/

	// Copy res to 2D-memory area for correct padding.
	float* res_array;
	size_t pitch;
	cuUtilsSafeCall( cudaMallocPitch<float>(&res_array, &pitch, obsH*sizeof(float), obsW) );
	cuUtilsSafeCall( cudaMemcpy2D(res_array, pitch, abs_buffer, sizeof(float)*obsH, sizeof(float)*obsH, obsW, cudaMemcpyDeviceToDevice) );
	//printf("%d\n", pitch);

	// Bind res to texture
	cudaChannelFormatDesc resultChannelDesc = cudaCreateChannelDesc<float>();
	response.addressMode[0] = cudaAddressModeClamp;
	response.addressMode[1] = cudaAddressModeClamp;
	response.filterMode = cudaFilterModeLinear;
	response.normalized = true;

	// bind resulting field to texture
	cuUtilsSafeCall( cudaBindTexture2D(NULL, response, (const void*) res_array, resultChannelDesc, obsH, obsW, pitch) );

	// map PBO to get CUDA device pointer	
	cuUtilsSafeCall( cudaGLMapBufferObject((void**)&d_pbo_buffer, pbo) );
	//size_t g_res_size = 0;
	//cuUtilsSafeCall( cudaGraphicsGLRegisterBuffer(&g_res, pbo, cudaGraphicsRegisterFlagsNone) );
	//cuUtilsSafeCall( cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_buffer, &g_res_size, *g_res) );
	cuUtilsSafeCall( cudaMemset(d_pbo_buffer, 0, numOfPixels()*4) );

	// Copy result to pbo using the texture above
	const dim3 block(16, 32, 1);
	const dim3 grid((dispDim.h-1)/block.x + 1, (dispDim.w-1)/block.y + 1);

	DisplayKernel<<<grid, block>>>(d_pbo_buffer, dispDim.w, dispDim.h, dynamicRange);

	cuUtilsSafeCall( cudaThreadSynchronize() );

	cuUtilsSafeCall( cudaGetLastError() );

	cuUtilsSafeCall( cudaFree(abs_buffer) );
	cuUtilsSafeCall( cudaFree(res_array) );

	// unmap PBO
	cuUtilsSafeCall( cudaGLUnmapBufferObject(pbo) ); 
	//cuUtilsSafeCall( cudaGraphicsUnregisterResource(g_res) );

	// draw processed buffer on display
	drawBuffer();
}

GLuint DisplayResponse::createPBO()
{
	if (pbo)
	{
		deletePBO();
	}

	// create pixel buffer object for display
	GLsizeiptr memSize = numOfPixels() * sizeof(GLubyte) * 4;
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, memSize, 0, GL_STREAM_DRAW_ARB);
	//glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cuUtilsSafeCall( cudaGLRegisterBufferObject(pbo) );
	return pbo;
}

GLuint DisplayResponse::getPBO()
{
	return pbo;
}

void DisplayResponse::deletePBO()
{
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	cuUtilsSafeCall( cudaGLUnregisterBufferObject(pbo) );    
	glDeleteBuffersARB(1, &pbo);
	pbo = 0;
}

void DisplayResponse::setDispDim(Dimension<int> dim)
{
	dispDim = dim;
	createPBO();
}

Dimension<int> DisplayResponse::getDispDim()
{
	return dispDim;
}

int DisplayResponse::getWidth()
{
	return dispDim.w;
}

void DisplayResponse::setWidth(int w)
{
	dispDim.w = w; 
	createPBO();
}
	
int DisplayResponse::getHeight()
{
	return dispDim.h;
}

void DisplayResponse::setHeight(int h)
{
	dispDim.h = h;
	createPBO();
}

