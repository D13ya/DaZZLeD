package crypto

import (
	"crypto/rand"
	"encoding/binary"
	"errors"

	"github.com/cloudflare/circl/oprf"
)

var oprfSuite = oprf.SuiteRistretto255

type OPRFClient struct {
	client oprf.Client
}

type OPRFServer struct {
	server oprf.Server
}

type OPRFState struct {
	finalize *oprf.FinalizeData
}

func NewOPRFClient() *OPRFClient {
	return &OPRFClient{client: oprf.NewClient(oprfSuite)}
}

func NewOPRFServer(privateKey *oprf.PrivateKey) *OPRFServer {
	return &OPRFServer{server: oprf.NewServer(oprfSuite, privateKey)}
}

func GenerateOPRFKeyPair() (privateKey []byte, publicKey []byte, err error) {
	key, err := oprf.GenerateKey(oprfSuite, rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	privateKey, err = key.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}
	publicKey, err = key.Public().MarshalBinary()
	if err != nil {
		return nil, nil, err
	}
	return privateKey, publicKey, nil
}

func ParseOPRFPrivateKey(data []byte) (*oprf.PrivateKey, error) {
	key := &oprf.PrivateKey{}
	if err := key.UnmarshalBinary(oprfSuite, data); err != nil {
		return nil, err
	}
	return key, nil
}

func ParseOPRFPublicKey(data []byte) (*oprf.PublicKey, error) {
	key := &oprf.PublicKey{}
	if err := key.UnmarshalBinary(oprfSuite, data); err != nil {
		return nil, err
	}
	return key, nil
}

func (c *OPRFClient) Blind(input []byte) (*OPRFState, []byte, error) {
	if c == nil {
		return nil, nil, errors.New("oprf client is nil")
	}
	fin, req, err := c.client.Blind([][]byte{input})
	if err != nil {
		return nil, nil, err
	}
	wire, err := marshalOPRFElements(req.Elements)
	if err != nil {
		return nil, nil, err
	}
	return &OPRFState{finalize: fin}, wire, nil
}

func (c *OPRFClient) Finalize(state *OPRFState, evalBytes []byte) ([]byte, error) {
	if c == nil || state == nil || state.finalize == nil {
		return nil, errors.New("oprf finalize state is nil")
	}
	eval, err := unmarshalEvaluation(evalBytes)
	if err != nil {
		return nil, err
	}
	outputs, err := c.client.Finalize(state.finalize, eval)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.New("unexpected oprf output count")
	}
	return outputs[0], nil
}

func (s *OPRFServer) Evaluate(requestBytes []byte) ([]byte, error) {
	if s == nil {
		return nil, errors.New("oprf server is nil")
	}
	req, err := unmarshalEvaluationRequest(requestBytes)
	if err != nil {
		return nil, err
	}
	eval, err := s.server.Evaluate(req)
	if err != nil {
		return nil, err
	}
	return marshalOPRFElements(eval.Elements)
}

func marshalOPRFElements(elements []oprf.Blinded) ([]byte, error) {
	if len(elements) > 0xFFFF {
		return nil, errors.New("oprf element count exceeds uint16")
	}
	elemLen := int(oprfSuite.Group().Params().CompressedElementLength)
	out := make([]byte, 2, 2+len(elements)*elemLen)
	binary.BigEndian.PutUint16(out, uint16(len(elements)))
	for _, element := range elements {
		blob, err := element.MarshalBinaryCompress()
		if err != nil {
			return nil, err
		}
		out = append(out, blob...)
	}
	return out, nil
}

func unmarshalOPRFElements(data []byte) ([]oprf.Blinded, error) {
	if len(data) < 2 {
		return nil, errors.New("oprf element payload too short")
	}
	count := int(binary.BigEndian.Uint16(data[:2]))
	elemLen := int(oprfSuite.Group().Params().CompressedElementLength)
	expected := 2 + count*elemLen
	if len(data) != expected {
		return nil, errors.New("oprf element payload length mismatch")
	}
	elements := make([]oprf.Blinded, count)
	offset := 2
	for i := 0; i < count; i++ {
		element := oprfSuite.Group().NewElement()
		if err := element.UnmarshalBinary(data[offset : offset+elemLen]); err != nil {
			return nil, err
		}
		elements[i] = element
		offset += elemLen
	}
	return elements, nil
}

func unmarshalEvaluationRequest(data []byte) (*oprf.EvaluationRequest, error) {
	elements, err := unmarshalOPRFElements(data)
	if err != nil {
		return nil, err
	}
	return &oprf.EvaluationRequest{Elements: elements}, nil
}

func unmarshalEvaluation(data []byte) (*oprf.Evaluation, error) {
	elements, err := unmarshalOPRFElements(data)
	if err != nil {
		return nil, err
	}
	return &oprf.Evaluation{Elements: elements, Proof: nil}, nil
}
