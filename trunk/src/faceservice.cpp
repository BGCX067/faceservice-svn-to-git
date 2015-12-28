#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/HTTPHTMLHeader.h"
#include "cgicc/HTMLClasses.h"

#include <stdio.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <curl/curl.h>

using namespace std;
using namespace cgicc;     // Or reference as cgicc::Cgicc formData; below in object instantiation.

const char* cascade_name = "/usr/local/share/faceservice/face_detect.xml";




// (setq c-basic-offset 4)
void error(int num, const char *msg)
{
    char buf[10];
    sprintf(buf, "%d", num);

    cout << HTTPContentHeader("text/xml");

    cout << "<err>" << endl;
    cout << "    <code>" << buf << "</code>" << endl;
    cout << "    <message>" << msg << "</message>" << endl;
    cout << "</err>" << endl;
    exit(1);
}


int getFaces(IplImage *image, CvRect **facelocs, double *t)
{
    //static double scale = 1200 / (double)image->width;
    static double scale = 1.0;
    //if (scale < 0) scale = 1.0;
    //static double scale =
   
    static int boxsize = (int)(image->width * 0.04); // heuristic
   
    static CvMemStorage* storage = 0;
    static CvHaarClassifierCascade* cascade = 0;
    
    IplImage *gray = 0, *small_image = 0;

    gray = cvCreateImage( cvSize(image->width, image->height), 8, 1 );
    small_image = cvCreateImage( cvSize( cvRound (image->width/scale), cvRound (image->height/scale)),8, 1 );

    if (!cascade) {
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
    
	if( !cascade ) {
	    error(99, "Installation problem: Could not load classifier cascade" );
	}
    }
    if (!storage) {
	storage = cvCreateMemStorage(0);
    }

    cvCvtColor( image, gray, CV_BGR2GRAY );
    cvResize( gray, small_image, CV_INTER_LINEAR );
    cvEqualizeHist( small_image, small_image );
    cvClearMemStorage( storage );

    *t = (double)cvGetTickCount();
    CvSeq* faces = cvHaarDetectObjects( small_image, cascade, storage,
					1.2, 2, 0,
					cvSize(boxsize, boxsize) );
    *t = (double)cvGetTickCount() - *t;
    
    if (faces->total <= 0)
	return 0; // no face found

    *facelocs = (CvRect *)malloc(faces->total * sizeof(CvRect));
    for (int i=0; i<faces->total; i++) {

	CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
	(*facelocs)[i].x =      (int)(r->x * scale); //  + r->width/2;
	(*facelocs)[i].y =      (int)(r->y * scale); // + r->height/2;
	(*facelocs)[i].width =  (int)(r->width * scale);
	(*facelocs)[i].height = (int)(r->height * scale);
    }

    return faces->total; // faces found
}


/////////////////////////////////////

// (setq c-basic-offset 4)

char *getFileFromRawPostData(CgiEnvironment env)
{
    // write file
    char *fname = tempnam("/tmp", "facesvc");
    
    ofstream out(fname, ios::out | ios::binary); 
    out.write(env.getPostData().c_str(), env.getPostData().length());
    out.close();
    
    return fname;
}

char *getFileFromUpload(const_file_iterator file)
{
    // write file
    char *fname = tempnam("/tmp", "facesvc");
    
    ofstream out(fname, ios::out | ios::binary); 
    file->writeToStream(out);
    out.close();
    
    return fname;
}

char * getFileFromURL(const char *url)
{
    static char errorBuffer[CURL_ERROR_SIZE];  

    // Our curl objects  
    CURL *curl;  
    CURLcode result;  

    char *fname = tempnam("/tmp", "facesvc");
    FILE *curl_fp = fopen(fname, "w");
    
    // Create our curl handle  
    curl = curl_easy_init(); 

    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer);  
    curl_easy_setopt(curl, CURLOPT_URL, url);  
    curl_easy_setopt(curl, CURLOPT_HEADER, 0);  
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);  
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, curl_fp); 

    // Attempt to retrieve the remote page  
    result = curl_easy_perform(curl);  

    curl_easy_cleanup(curl); 

    fclose(curl_fp);
   
    if (result == CURLE_OK) {
	return fname;
    }
    else {
	error(100+result, errorBuffer);
    }
	

}

void outputResponseXML(IplImage *img, int n, CvRect *faces)
{
    char buf[1024];

   cout << "<resp>" << endl;

   sprintf(buf, "<image width=\"%d\" height=\"%d\" />", img->width, img->height);
   cout << "  " << buf << endl;
   
   cout << "  <faces>" << endl;  
   for (int i=0; i<n; i++) {
      sprintf(buf, "<face x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" />",
	      faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      cout << "    " << buf << endl;
   }
   cout << "  </faces>" << endl;  
   cout << "</resp>" << endl;
}

void outputResponseJSON(IplImage *img, int n, CvRect *faces)
{
   char buf[1024];

   cout << "{" << endl;

   sprintf(buf, "image: {width:%d, height:%d},", img->width, img->height);
   cout << "  " << buf << endl;
   
   cout << "faces: [" << endl;  
   for (int i=0; i<n; i++) {
      sprintf(buf, "{x:%d, y:%d, width:%d, height:%d}",
	      faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      cout << "    " << buf;
      if (i < n-1) cout << "," << endl;;
   }
   cout << "  ]" << endl;  
   cout << "}" << endl;
}

int main(int argc, char **argv)
{
    try {
	Cgicc form;
	CgiEnvironment env = form.getEnvironment();

	if (env.getRequestMethod() != "POST") {
	    error(1, "Wrong request method.");
	}

	char *fname;

	if(env.getContentType().substr(0, 6) == "image/") {
	    // A raw image was posted
	    fname = getFileFromRawPostData(env);

	} else if (env.getContentType().substr(0,33) == "application/x-www-form-urlencoded") {
	    // Posted a www-form; expect the data to be in the 'url' field.
	    form_iterator url_value = form.getElement("image_url");

	    if( url_value != (*form).end() && !url_value->isEmpty()) {
		fname = getFileFromURL((**url_value).c_str());

	    } else {
		error(3, "Expecting 'image_url' field to be posted");
	    }
	} else if (env.getContentType().substr(0,19) == "multipart/form-data") {

	    const_file_iterator file = form.getFile("image_file");
	    if(file != form.getFiles().end()) {
		fname = getFileFromUpload(file);
	    } else { 
		error(3, "Expecting 'image_file' to be posted");
	    }

	    
	} else if (env.getContentType() == "application/url") {
	    // 'URL' content type
	    fname = getFileFromURL(env.getPostData().c_str());

	} else {
	    error(2, "Wrong content type, you must upload an image.");
	}

	CvRect *faces;
	double t;

	IplImage *img = cvLoadImage(fname);
        if (!img || !img->width || !img->height) {	    
	    error(4, "Image could not be loaded.");
	}
       
	int n = getFaces(img, &faces, &t);

       form_iterator fmt = form.getElement("format");

       if( fmt != (*form).end() && !fmt->isEmpty()) {
	  if ((**fmt) == "json") {
	     cout << HTTPContentHeader("text/javascript");
	     outputResponseJSON(img, n, faces);
	  }
	  else if ((**fmt) == "xml") {
	     cout << HTTPContentHeader("text/xml");       
	     outputResponseXML(img, n, faces);
	  }
	  else {
	     error(5, "Invalid value for 'format' parameter");
	  }
       }
       else {
	  // defaut format = XML
	  cout << HTTPContentHeader("text/xml");       
	  outputResponseXML(img, n, faces);
       }

       cvReleaseImage(&img);
       unlink(fname);
       free(faces);
       free(fname);
    }
    catch(exception& e) {

	error(99, "A generic error occurred.");
    }
    return 0;   // To avoid Apache errors.
}
