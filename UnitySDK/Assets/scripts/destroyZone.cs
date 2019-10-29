using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class destroyZone : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
    private void OnTriggerEnter2D(Collider2D collision)
    {
        //if (collision.gameObject.tag == "redNotes"
        //||collision.gameObject.tag == "blueNotes"
        //    || collision.gameObject.tag == "Notes") {
        //    Destroy(collision.gameObject);
        //}
    }
}
