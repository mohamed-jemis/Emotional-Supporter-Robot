import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final picker = ImagePicker();
  late File _image = File(''); // Provide a default or initial value
  bool _loading = false;
  late List<dynamic> _output = []; // Provide a default or initial value

  @override
  void initState() {
    super.initState();
    _loading = true;
    loadModel().then((value) {
      // TODO: add some interactivity
    });
  }
  @override void dispose() {
    super.dispose();
    Tflite.close();
  }

  pickImage() async {
    var image = await picker.getImage(source: ImageSource.camera);
    if (image == null) return null;
    setState(() {
      _image = File(image.path);
    });
    classifyImage(_image);
  }

  pickGalleryImage() async {
    var image = await picker.getImage(source: ImageSource.gallery);
    if (image == null) return null;
    setState(() {
      _image = File(image.path);
    });
    classifyImage(_image);
  }

  classifyImage(File image) async {
    var output = await Tflite.runModelOnImage(
        path: image.path,
        numResults: 2,
        threshold: 0.5,
        imageMean: 127.5,
        imageStd: 127.5);
    setState(() {
      _loading = false;
      _output = output ?? [];
    });
  }
  loadModel() async {
    await Tflite.loadModel(
      model: 'assets/FER.tflite',
      labels: 'assets/emotions.txt',
    );
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Emotion"),
      ),
      body: Container(
        padding: EdgeInsets.only(left: 10, right: 10),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              // Buttons
              Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  _GroupText('Choose source:'),
                  ButtonBar(
                    alignment: MainAxisAlignment.center,
                    buttonMinWidth: 150,
                    layoutBehavior: ButtonBarLayoutBehavior.padded,
                    buttonPadding: EdgeInsets.symmetric(vertical: 10),
                    children: <Widget>[
                      ElevatedButton(
                        onPressed: pickImage,
                        child: Text('Cam'),
                      ),
                      SizedBox(width: 20),
                      ElevatedButton(
                        onPressed: pickGalleryImage,
                        child: Text('Gallery'),
                      ),
                    ],
                  )
                ],
              ),
              _SpaceLine(),
              // Image
              Center(
                child: _loading ?
                Container(
                  width: 300,
                  child: Column(
                    children: <Widget>[
                      SizedBox(height: 50),
                      Image.asset('assets/room.png'),
                    ],
                  ),
                )
                    : Container(
                    child: Column(
                      children: <Widget>[
                        _output != null ?
                        Container(
                          padding: EdgeInsets.symmetric(vertical: 10),
                          child: Text('${_output[0]['label']}',
                              style: TextStyle(
                                  color: Colors.red, fontSize: 20.0)),
                        )
                            : Container(),
                        SizedBox(height: 20,),
                        Container(
                          height: 250,
                          child: Image.file(_image),
                        ),
                      ],
                    )),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
class _GroupText extends StatelessWidget {
  final String text;
  const _GroupText(this.text);
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 5, horizontal: 15),
      child: Text(
        text,
        style: TextStyle(fontSize: 25, fontWeight: FontWeight.w500),
      ),
    );
  }
}
class _SpaceLine extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 5,
      child: Container(
        color: Colors.grey,
      ),
    );
  }
}
// class _HomeState extends State<Home> {
//   bool _loading = true;
//   late File _image;
//   late List _output;
//   final picker = ImagePicker(); //allows us to pick image from gallery or camera
//
//   @override
//   void initState() {
//     //initS is the first function that is executed by default when this class is called
//     super.initState();
//     // loadModel().then((value) {
//     //   setState(() {});
//     // });
//   }
//
//   @override
//   void dispose() {
//     //dis function disposes and clears our memory
//     super.dispose();
//     // Tflite.close();
//   }
//
//   // classifyImage(File image) async {
//   //   //this function runs the model on the image
//   //   var output = await Tflite.runModelOnImage(
//   //     path: image.path,
//   //     numResults:
//   //     5, //the amout of categories our neural network can predict (here no. of animals)
//   //     threshold: 0.5,
//   //     imageMean: 127.5,
//   //     imageStd: 127.5,
//   //   );
//   //   setState(() {
//   //     _output = output!;
//   //     _loading = false;
//   //   });
//   // }
//
//   // loadModel() async {
//   //   //this function loads our model
//   //   await Tflite.loadModel(
//   //     model: 'assets/model_unquant.tflite',
//   //     labels: 'assets/labels.txt',
//   //   );
//   // }
//
//   pickImage() async {
//     //this function to grab the image from camera
//     var image = await picker.pickImage(source: ImageSource.camera);
//     if (image == null) return null;
//
//     setState(() {
//       _image = File(image.path);
//     });
//     // classifyImage(_image);
//   }
//
//   pickGalleryImage() async {
//     //this function to grab the image from gallery
//     var image = await picker.pickImage(source: ImageSource.gallery);
//     if (image == null) return null;
//
//     setState(() {
//       _image = File(image.path);
//     });
//     // classifyImage(_image);
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         backgroundColor: Colors.indigo,
//         centerTitle: true,
//         title: Text(
//           'EIR App',
//           style: TextStyle(
//             color: Colors.white,
//             fontWeight: FontWeight.w500,
//             fontSize: 23,
//           ),
//         ),
//       ),
//       body: Container(
//         color: Color.fromRGBO(68, 190, 255, 0.8),
//         padding: EdgeInsets.symmetric(horizontal: 35, vertical: 50),
//         child: Container(
//           alignment: Alignment.center,
//           padding: EdgeInsets.all(30),
//           decoration: BoxDecoration(
//             color: Colors.indigo,
//             borderRadius: BorderRadius.circular(30),
//           ),
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               Container(
//                 child: Center(
//                   child: _loading == true
//                       ? null //show nothing if no picture selected
//                       : Container(
//                     child: Column(
//                       children: [
//                         Container(
//                           height: MediaQuery.of(context).size.width * 0.5,
//                           width: MediaQuery.of(context).size.width * 0.5,
//                           child: ClipRRect(
//                             borderRadius: BorderRadius.circular(30),
//                             child: Image.file(
//                               _image,
//                               fit: BoxFit.fill,
//                             ),
//                           ),
//                         ),
//                         Divider(
//                           height: 25,
//                           thickness: 1,
//                         ),
//                         // ignore: unnecessary_null_comparison
//                         _output != null
//                             ? Text(
//                           'The animal is: ${_output[0]['label']}',
//                           style: TextStyle(
//                             color: Colors.white,
//                             fontSize: 18,
//                             fontWeight: FontWeight.w400,
//                           ),
//                         )
//                             : Container(),
//                         Divider(
//                           height: 25,
//                           thickness: 1,
//                         ),
//                       ],
//                     ),
//                   ),
//                 ),
//               ),
//               Container(
//                 child: Column(
//                   children: [
//                     GestureDetector(
//                       onTap: pickImage,
//                       child: Container(
//                         width: MediaQuery.of(context).size.width - 200,
//                         alignment: Alignment.center,
//                         padding:
//                         EdgeInsets.symmetric(horizontal: 24, vertical: 17),
//                         decoration: BoxDecoration(
//                           color: Colors.blue,
//                           borderRadius: BorderRadius.circular(15),
//                         ),
//                         child: Text(
//                           'Take A Photo',
//                           style: TextStyle(color: Colors.white, fontSize: 16),
//                         ),
//                       ),
//                     ),
//                     SizedBox(height: 30),
//                     GestureDetector(
//                       onTap: pickGalleryImage,
//                       child: Container(
//                         width: MediaQuery.of(context).size.width - 200,
//                         alignment: Alignment.center,
//                         padding:
//                         EdgeInsets.symmetric(horizontal: 24, vertical: 17),
//                         decoration: BoxDecoration(
//                           color: Colors.blue,
//                           borderRadius: BorderRadius.circular(15),
//                         ),
//                         child: Text(
//                           'Pick From Gallery',
//                           style: TextStyle(color: Colors.white, fontSize: 16),
//                         ),
//                       ),
//                     ),
//                   ],
//                 ),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }