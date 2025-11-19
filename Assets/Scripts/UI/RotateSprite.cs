//
// Copyright 2020
// Author: Martin Smith
//

using UnityEngine;

namespace Trace.UI
{
	public class RotateSprite : MonoBehaviour
	{
		[SerializeField] private float _degreesPerSecond = 90;

		public float DegreesPerSecond {
			get => _degreesPerSecond;
			set => _degreesPerSecond = value;
		}

		void Update()
		{
			transform.Rotate(Vector3.forward, Time.deltaTime * _degreesPerSecond, Space.Self);
		}
	}
}
