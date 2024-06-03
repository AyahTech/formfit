import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ImageBackground } from 'react-native';
import { Camera } from 'expo-camera';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useNavigation } from '@react-navigation/native';
import { Ionicons, AntDesign } from '@expo/vector-icons'; // Import AntDesign for icons

const ImproveForm = () => {
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraOn, setCameraOn] = useState(false); // Track if the camera is on or off
  const [cameraPermission, setCameraPermission] = useState(null);
  const navigation = useNavigation(); // Get the navigation object

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  const requestCameraPermission = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    setCameraPermission(status === 'granted');
    setCameraOn(status === 'granted'); // Automatically turn on the camera if permission is granted
  };

  if (cameraPermission === null) {
    return (
      <View style={styles.container}>
        <TouchableOpacity style={styles.crossButton} onPress={() => navigation.navigate('Exercise')}>
        <AntDesign name="closecircle" size={28} color="black" />
        </TouchableOpacity>
        <Text style={styles.headingText}>Get real-time form feedback to build injury-free exercise routine.</Text>
        <Text style={styles.subtext}>
        ⚠️ This module is currently under development. We are encountering issues related to model size and associated costs. The functionality was demonstrated in a presentation using a demo running on a local machine.
        </Text>
        <View style={styles.avatarContainer}>
          <ImageBackground source={require('../assets/images/scanner.png')} style={styles.avatar} />
        </View>
        <TouchableOpacity style={styles.button} onPress={requestCameraPermission}>
          <Text style={styles.buttonText}>Enable Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (cameraPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.headingText}>Camera permission denied</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.crossButton} onPress={() => navigation.navigate('Exercise')}>
        <Ionicons name="close" size={24} color="white" />
      </TouchableOpacity>

      {cameraOn && (
        <>
          <CameraView style={styles.camera} facing={facing}>
            <View style={styles.buttonContainer}>
              <TouchableOpacity style={styles.iconButton} onPress={toggleCameraFacing}>
                <Ionicons name="camera-reverse" size={35} color="white" />
              </TouchableOpacity>
            </View>
          </CameraView>
          <TouchableOpacity style={styles.headerLeftButton} onPress={toggleCamera}>
            <AntDesign name="closecircleo" size={35} color="white" />
          </TouchableOpacity>
        </>
      )}

      {!cameraOn && (
        <View style={styles.container}>
          <TouchableOpacity style={styles.crossButton} onPress={() => navigation.navigate('Exercise')}>
            <Ionicons name="close" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headingText}>Get real-time form feedback to build injury-free exercise routine.</Text>
          <Text style={styles.subtext}>
            Enable access to camera and microphone to receive real-time exercise tips via audio. Your video and audio are
            only accessed while on the "Improve Form" screen.
          </Text>
          <View style={styles.avatarContainer}>
            <View style={styles.avatar} />
          </View>
          <TouchableOpacity style={styles.button} onPress={requestCameraPermission}>
            <Text style={styles.buttonText}>Enable Access</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }

  function toggleCamera() {
    if (cameraOn) {
      setCameraPermission(null);
    }
    setCameraOn((prevState) => !prevState); // Toggle the camera state
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
  },
  crossButton: {
    position: 'absolute',
    top: 20,
    left: 6,
    borderRadius: 15, // Make the button circular
    padding: 10, // Adjust the padding to make the button smaller
  },
  headingText: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 60,
    marginBottom: 16,
  },
  subtext: {
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 24,
    fontWeight: 'bold',
    color:'red'
  },
  avatarContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: 20,
  },
  avatar: {
    width: 353,
    height: 353,
    backgroundColor: '#ccc',
    marginBottom:50,
  },
  button: {
    backgroundColor: 'blue',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 4,
    position: 'absolute',
    bottom: 30,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  camera: {
    flex: 1,
    width: 390,
    
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'flex-end', // Align the buttons at the bottom
    marginBottom: 20, // Add some margin at the bottom
  },
  iconButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 50,
    padding: 10,
  },
  headerLeftButton: {
    position: 'absolute',
    top: 10,
    left: 20, // Position it slightly to the right of the crossButton
  },
  closeCameraButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});

export default ImproveForm;
