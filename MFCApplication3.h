
// MFCApplication3.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// CMFCApplication3App:
// �аѾ\��@�����O�� MFCApplication3.cpp
//

class CMFCApplication3App : public CWinApp
{
public:
	CMFCApplication3App();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern CMFCApplication3App theApp;